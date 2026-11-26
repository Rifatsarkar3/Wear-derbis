"""Proposed mechanism-linked multi-view particle graph model (DESIGN sec. 7).

particle (view graphs -> GNN -> view pooling, crop CNN, EDS, handcrafted globals) -> mechanism + volume heads ->
bag aggregation per oil sample -> severity (CORAL) + interval rate -> run embedding -> run rate, with the aggregate
debris-volume physics of DESIGN sec. 3 computed in float32 outside autocast.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from wearpg.batch import ModelDims, RunBatch
from wearpg.models.heads import CoralHead, mlp
from wearpg.models.image_branch import make_image_branch
from wearpg.models.particle_gnn import ParticleGNN, pool_views
from wearpg.models.physics import debris_log_volume_run, debris_log_volume_sample, metal_fraction_log
from wearpg.models.run_aggregator import MeanPool, make_aggregator
from wearpg.schema import MECHANISMS

LN10 = math.log(10.0)
# Initial output of mlp-only rate heads (physics_head=false): log10 1e-5 mm^3/(N m), a mid-range metallic specific
# wear rate, so that ablation does not spend its training budget drifting from 0 to the target scale.
LOG10_K_INIT = -5.0


class MechanismLinkedModel(nn.Module):
    """Honours every `model.*` flag of configs/default.yaml; graph-selection flags (edges, use_interior_nodes,
    use_fractal_features, use_multiview) act through the input dims in `dims` / the batch only."""

    def __init__(self, cfg: dict, dims: ModelDims):
        super().__init__()
        m = cfg["model"]
        h, p = int(m["hidden"]), float(m["dropout"])
        self.use_composition = bool(m["use_composition"])
        self.use_conditions = bool(m["use_operating_conditions"])
        self.chunk = int(m["gnn_chunk_graphs"])
        self.gnn = (ParticleGNN(dims.node_dim, dims.edge_dim, h, int(m["gnn_layers"]), m["conv"], p, tuple(m["pooling"]))
                    if m["use_graph"] else None)
        self.image = (make_image_branch(m["image_backbone"], dims.crop_channels, h, bool(m["image_pretrained"]),
                                        dims.crop_size) if m["use_texture_branch"] else None)
        in_dim = ((2 * self.gnn.out_dim if self.gnn is not None else 0) + (h if self.image is not None else 0)
                  + (dims.n_elements + 1 if self.use_composition else 0) + dims.particle_feat_dim)
        self.particle_mlp = mlp(in_dim, h, h, p)
        self.mech_head = nn.Linear(h, dims.n_classes)
        # mechanism_link=false (ablation no_mechanism_link): mechanism-agnostic volume, s(ln A) instead of s(z) and
        # metal fraction 1 for every class
        self.mechanism_link = bool(m["mechanism_link"])
        self.volume_head = nn.Linear(h if self.mechanism_link else 1, 1)  # starts at ln 0.3, i.e. v = 0.3 A^1.5
        nn.init.zeros_(self.volume_head.weight)
        nn.init.constant_(self.volume_head.bias, math.log(0.3))

        self.aggregator = make_aggregator(m["aggregator"], h, h) if m["use_run_aggregation"] else None
        self.sample_mlp = mlp(h + dims.sample_feat_dim, h, h, p) if self.aggregator is not None else None
        self.severity_head = CoralHead(h, dims.n_severity)
        self.rate_sample_mlp = mlp(h, h, 1, p)
        self.run_mlp = mlp(h + (dims.cond_dim if self.use_conditions else 0), h, h, p)
        self.rate_run_mlp = mlp(h, h, 1, p)
        physics_head = bool(m["physics_head"])
        self.a_sample = nn.Parameter(torch.tensor(1.0)) if physics_head else None
        self.a_run = nn.Parameter(torch.tensor(1.0)) if physics_head else None
        if not physics_head:
            for head in (self.rate_sample_mlp, self.rate_run_mlp):
                nn.init.constant_(head[-1].bias, LOG10_K_INIT)

        self.theta = nn.Parameter(torch.full((dims.n_magnetic,), float(m["init_log_eta"])))
        self.delta = nn.Parameter(torch.zeros(dims.n_materials)) if m["per_material_eta"] else None
        mf = cfg["physics"]["metal_fraction"]
        if set(mf) - set(MECHANISMS):
            raise ValueError(f"physics.metal_fraction has unknown classes {sorted(set(mf) - set(MECHANISMS))}")
        self.register_buffer("c_by_class", torch.tensor([float(mf.get(c, 1.0)) if self.mechanism_link else 1.0
                                                         for c in MECHANISMS]))

    def _embed_views(self, g) -> Tensor:
        """(n_v, gnn.out_dim) view embeddings; without autograd at most `gnn_chunk_graphs` graphs per GNN call."""
        n = int(g.num_graphs)
        if torch.is_grad_enabled() or n <= self.chunk:
            return self.gnn(g.x, g.edge_index, g.edge_attr, g.batch, n)
        ptr = torch.cat([g.batch.new_zeros(1), torch.bincount(g.batch, minlength=n).cumsum(0)]).tolist()
        src, out = g.edge_index[0], []
        for g0 in range(0, n, self.chunk):  # PyG batches keep each graph's nodes contiguous and in graph order
            g1 = min(g0 + self.chunk, n)
            n0, n1 = ptr[g0], ptr[g1]
            e = (src >= n0) & (src < n1)
            out.append(self.gnn(g.x[n0:n1], g.edge_index[:, e] - n0, g.edge_attr[e], g.batch[n0:n1] - g0, g1 - g0))
        return torch.cat(out)

    @staticmethod
    def _physics_rate(x: Tensor, resid: Tensor, a: Tensor | None) -> Tensor:
        """a * x + resid where x is finite, else NaN; x is zero-filled first so masked entries give `a` no NaN grad."""
        if a is None:
            return resid
        ok = torch.isfinite(x)
        return torch.where(ok, a * torch.where(ok, x, 0.0) + resid, float("nan"))

    def forward(self, batch: RunBatch) -> dict[str, Tensor]:
        """Outputs (DESIGN sec. 7): particle_emb, mech_logits (n_p, C), log_volume (n_p,) [ln um^3], sample_emb,
        severity_logits (n_s, K-1), log10_rate_sample (n_s,) (NaN for cumulative runs), run_emb, log10_rate_run,
        log_eta_run, log_debris_volume_sample_mm3, log_debris_volume_run_mm3, coverage_run (n_r,), attention (n_p,),
        eta_offset_l2 (). Runs without observed samples get NaN debris volume / physics rates and coverage 0."""
        b = batch
        n_p, n_s, n_r = b.n_particles, b.n_samples, b.n_runs
        p2s, s2r = b.particle_to_sample, b.sample_to_run

        parts = []
        if self.gnn is not None:
            parts.append(pool_views(self._embed_views(b.graphs), b.view_to_particle, n_p))
        if self.image is not None:
            if b.crops is None:
                raise ValueError("model.use_texture_branch=true needs RunBatch.crops, got None")
            parts.append(self.image(b.crops))
        if self.use_composition:
            known = b.composition_mask[:, None]
            parts += [torch.where(known, b.composition, 0.0), known.float()]
        parts.append(b.particle_feats)
        z = self.particle_mlp(torch.cat(parts, 1))
        mech_logits = self.mech_head(z).float()

        if self.aggregator is not None:
            pooled, attention = self.aggregator(z, p2s, n_s)
            sample_emb = self.sample_mlp(torch.cat([pooled, b.sample_feats], 1))
            severity_logits = self.severity_head(sample_emb)
            rate_s = self.rate_sample_mlp(sample_emb).squeeze(-1)
        else:  # late fusion: per-particle predictions averaged per sample
            sample_emb, attention = MeanPool()(z, p2s, n_s)
            severity_logits = scatter(self.severity_head(z), p2s, 0, n_s, "mean")
            rate_s = scatter(self.rate_sample_mlp(z).squeeze(-1), p2s, 0, n_s, "mean")
        run_in = scatter(sample_emb, s2r, 0, n_r, "mean")  # zeros for a run without observed samples
        if self.use_conditions:
            run_in = torch.cat([run_in, b.run_cond_feats], 1)
        run_emb = self.run_mlp(run_in)
        rate_r = self.rate_run_mlp(run_emb).squeeze(-1)

        with torch.autocast(z.device.type, enabled=False):
            vol_in = z.float() if self.mechanism_link else b.log_area_um2.float()[:, None]
            log_volume = 1.5 * b.log_area_um2.float() + self.volume_head(vol_in).squeeze(-1)
            log_eta_run = self.theta[b.magnetic_idx] + (self.delta[b.material_idx] if self.delta is not None else 0.0)
            log_c = metal_fraction_log(torch.softmax(mech_logits, -1).detach(), self.c_by_class)
            log_d_s = debris_log_volume_sample(log_volume, log_c, p2s, n_s, b.log_subsample_factor,
                                               b.log_obs_fraction, log_eta_run[s2r])
            log_v_r, coverage = debris_log_volume_run(log_d_s, s2r, n_r, b.sample_distance_m, b.sliding_distance_m,
                                                      b.sample_time_s, b.sample_log_q, b.run_cumulative,
                                                      b.sample_distance_end_m)
            log_f = torch.log(b.load_N.float())
            x_s = (log_d_s - log_f[s2r] - torch.log(b.sample_distance_m.float())) / LN10  # log10 V_s / (F ds_s)
            x_r = (log_v_r - log_f - torch.log(b.sliding_distance_m.float())) / LN10    # log10 V_r / (F s)
            rate_s = self._physics_rate(x_s, rate_s.float(), self.a_sample)
            rate_s = torch.where(b.run_cumulative[s2r], float("nan"), rate_s)  # no sample-level debris rate
            rate_r = self._physics_rate(x_r, rate_r.float(), self.a_run)

        return {
            "particle_emb": z, "mech_logits": mech_logits, "log_volume": log_volume,
            "sample_emb": sample_emb, "severity_logits": severity_logits.float(), "log10_rate_sample": rate_s,
            "run_emb": run_emb, "log10_rate_run": rate_r, "log_eta_run": log_eta_run,
            "log_debris_volume_sample_mm3": log_d_s, "log_debris_volume_run_mm3": log_v_r, "coverage_run": coverage,
            "attention": attention,
            "eta_offset_l2": self.delta.pow(2).sum() if self.delta is not None else self.theta.new_zeros(()),
        }

    @torch.no_grad()
    def fill_unseen_eta(self, seen_material_idx: list[int]) -> None:
        """delta = 0 for materials absent from training, so they use their magnetic class theta only."""
        if self.delta is not None:
            unseen = torch.ones_like(self.delta, dtype=torch.bool)
            unseen[torch.as_tensor(list(seen_material_idx), dtype=torch.long, device=unseen.device)] = False
            self.delta[unseen] = 0.0

    def eta_transferable(self, material_idx: int, seen_material_idx, magnetic_of_material) -> bool:
        """True iff a training material shares the magnetic-capture class of `material_idx` (so theta was fitted).
        `magnetic_of_material[m]` gives material m's class (list, array, tensor or dict)."""
        cls = int(magnetic_of_material[material_idx])
        return any(int(magnetic_of_material[s]) == cls for s in seen_material_idx)
