"""Reading 1 of the hold-out family (exact vector positions, axes from grid lines). Writes reading1_part5.json and prints D and W per row."""
import json
from pathlib import Path

OUT = Path(__file__).with_name('reading1_part5.json')


def lin(v, p0, v0, p1, v1):
    return v0 + (v - p0) / (p1 - p0) * (v1 - v0)


# ---- jet495: Fig. 7 cumulative CPUC, y grid 0 at 700.78, 1000 at 586.75; marker centres (page 9)
y7 = lambda y: lin(y, 700.78, 0, 586.75, 1000)
jet_cum = {'M1': [y7(v) for v in (693.70, 684.60, 678.10, 662.85)], 'E1': [y7(v) for v in (680.53, 654.23, 632.43, 600.47)], 'E2': [y7(v) for v in (691.30, 669.50, 651.65, 624.20)]}
jet_W = {'M1': [3 + 0, 6 + 4, 5 + 0], 'E1': [11 + 12, 12 + 9, 12 + 14], 'E2': [6 + 7, 5 + 2, 18 + 13]}       # data labels of Fig. 5 (pinion + wheel)
# ---- tribint2009: Fig. 9 cumulative CPUC against cumulative mass loss; y 0 at 710.54, 600 at 569.11; x 0 at 65.16, 75 at 278.17
y9 = lambda y: lin(y, 710.54, 0, 569.11, 600)
tri_cum = {'M1': [y9(v) for v in (691.85, 677.55, 635.44, 583.80)], 'E1': [y9(v) for v in (685.88, 654.11, 622.33, 593.73)], 'E2': [y9(v) for v in (683.50, 665.24, 628.69, 595.33)]}
tri_W = {'M1': [4 + 1, 1 + 9, 16 + 11], 'E1': [4 + 5, 6 + 8, 12 + 29], 'E2': [1 + 6, 5 + 7, 14 + 11]}               # data labels of Fig. 7
# ---- ilt2011: Fig. 6 cumulative mass loss against cumulative CPUC at the end of each stage; x 0 at 105.69, 500 at 263.93; y 0 at 478.45, 100 at 329.66
x6 = lambda x: lin(x, 105.69, 0, 263.93, 500)
w6 = lambda y: lin(y, 478.45, 0, 329.66, 100)
ilt_c = [x6(v) for v in (128.12, 159.12, 204.96, 249.45)]
ilt_w = [w6(v) for v in (467.92, 433.27, 405.42, 339.51)]
# ---- pij2011: Fig. 7 CPUC at 12, 24, 36, 48 h; y 0 at 211.73, 100 at 110.78; labels of Fig. 6
y7p = lambda y: lin(y, 211.73, 0, 110.78, 100)
pij_c = {'P1': [y7p(v) for v in (167.99, 154.02, 130.21, 121.40)], 'P4': [y7p(v) for v in (192.56, 148.56, 143.95, 135.14)], 'P3': [y7p(v) for v in (207.08, 202.94, 203.96, 200.87)]}
pij_W = {'P1': 11 + 5, 'P4': 7 + 4, 'P3': 0}

rows = []
for oil in ('M1', 'E1', 'E2'):
    c = jet_cum[oil]
    d = [c[1], c[2] - c[1], c[3] - c[2]]
    for k, (st, w) in enumerate(zip(('K3+K6', 'K8', 'K9'), jet_W[oil])):
        rows.append(('jet495', oil, st, round(d[k], 2), w))
for oil in ('M1', 'E1', 'E2'):
    c = tri_cum[oil]
    d = [c[1], c[2] - c[1], c[3] - c[2]]
    for k, (st, w) in enumerate(zip(('K3+K6', 'K8', 'K9'), tri_W[oil])):
        rows.append(('tribint2009', oil, st, round(d[k], 2), w))
d = [ilt_c[0], ilt_c[1] - ilt_c[0], ilt_c[2] - ilt_c[1], ilt_c[3] - ilt_c[2]]
w = [ilt_w[0], ilt_w[1] - ilt_w[0], ilt_w[2] - ilt_w[1], ilt_w[3] - ilt_w[2]]
for k, st in enumerate(('K3', 'K6', 'K8', 'K9')):
    rows.append(('ilt2011', 'E2', st, round(d[k], 2), round(w[k], 2)))
for oil in ('P1', 'P4', 'P3'):
    c = pij_c[oil]
    rows.append(('pij2011', oil, 'end', round(c[3] + 40 / 1500 * sum(c[:3]), 2), pij_W[oil]))
res = {'cumulative': {'jet495': jet_cum, 'tribint2009': tri_cum, 'ilt2011': {'cpuc': ilt_c, 'mass_mg': ilt_w}, 'pij2011': pij_c}, 'rows': rows}
OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
for r in rows:
    print(r)
print({k: [round(v, 1) for v in x] for k, x in jet_cum.items()})
print({k: [round(v, 1) for v in x] for k, x in tri_cum.items()})
print([round(v, 1) for v in ilt_c], [round(v, 1) for v in ilt_w])
print({k: [round(v, 1) for v in x] for k, x in pij_c.items()})
