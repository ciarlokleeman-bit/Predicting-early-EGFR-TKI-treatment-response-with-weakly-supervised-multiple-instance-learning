"""Attention-based multiple instance learning on H&E whole-slide images.

Single-stream pipeline: tissue segmentation -> 512 px tiles at 0.5 um/px ->
Macenko normalisation -> frozen ImageNet encoder -> gated attention pooling ->
two-layer MLP -> sigmoid early poor-response score.
"""

__version__ = "0.3.2"
