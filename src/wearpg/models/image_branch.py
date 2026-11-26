"""Image (texture) branch on particle crops (DESIGN sec. 7)."""

from __future__ import annotations

from torch import Tensor, nn


class SmallCNN(nn.Module):
    """4 x [conv3x3 - GroupNorm - ReLU - maxpool2] (32-64-128-128 ch) -> global average pool -> Linear.

    Input (n, in_ch, H, W) float [0, 1] with H, W >= 16; output (n, out_dim). GroupNorm keeps the embedding
    independent of which particles share a batch (bags are small and correlated).
    """

    def __init__(self, in_ch: int = 4, out_dim: int = 128):
        super().__init__()
        layers, c = [], in_ch
        for co in (32, 64, 128, 128):
            layers += [nn.Conv2d(c, co, 3, padding=1), nn.GroupNorm(8, co), nn.ReLU(), nn.MaxPool2d(2)]
            c = co
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Linear(c, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.features(x))


def make_image_branch(name: str = "small_cnn", in_ch: int = 4, out_dim: int = 128, pretrained: bool = False,
                      img_size: int = 64) -> nn.Module:
    """`small_cnn` or any timm model id (e.g. resnet18, convnext_tiny) -> module mapping (n, in_ch, H, W) -> (n, out_dim).

    timm models use `num_classes=0` (pooled features) + a Linear projection; `img_size` is passed to ViT ids only
    (other timm constructors reject it).
    """
    if name == "small_cnn":
        return SmallCNN(in_ch, out_dim)
    import timm  # lazy: slow import, only needed for timm backbones

    kw = {"img_size": img_size} if name.startswith("vit") else {}
    net = timm.create_model(name, pretrained=pretrained, in_chans=in_ch, num_classes=0, **kw)
    return nn.Sequential(net, nn.Linear(net.num_features, out_dim))
