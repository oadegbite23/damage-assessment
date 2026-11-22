"""
Siamese U-Net for building localization + damage classification.

How it works
------------
One pretrained encoder (shared weights) runs on the pre-disaster and the
post-disaster image. Two decoders then predict:

    1. Localization: building / no building, decoded from the PRE image's
       features (footprints are visible before the disaster).
    2. Damage: 4 classes (no-damage, minor, major, destroyed), decoded from
       features that FUSE the two images at every scale, so the decoder
       sees what changed between them.

Outputs are raw logits (apply sigmoid / softmax or use `predict`):
    loc : (B, 1, H, W)
    dmg : (B, 4, H, W)   channel 0..3  <->  damage class 1..4

Damage classes are 1..4 in the masks, with 0 = background. The damage head
has only 4 channels: it is trained and scored on building pixels only, so
"background" is not one of its classes (the training loss ignores pixels
whose damage label is 0, including "un-classified" buildings).

Only the encoder comes from segmentation-models-pytorch. The decoders are
written out here (short) so the code doesn't depend on that library's
decoder API, which changes between versions.

Inputs must be float tensors, normalized with IMAGENET_MEAN / IMAGENET_STD,
with height and width divisible by 32 (512x512 crops are fine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder

N_DAMAGE_CLASSES = 4

# Pretrained ImageNet encoders expect inputs normalized with these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample 2x, concatenate the skip connection (if any), two convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    """Standard U-Net decoder.

    Takes the encoder's five feature maps ordered shallow -> deep
    (strides 2, 4, 8, 16, 32) and returns a full-resolution map with
    decoder_channels[-1] channels.
    """

    def __init__(self, enc_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        if len(enc_channels) != len(decoder_channels):
            raise ValueError("need one decoder stage per encoder feature map")
        skip_chs = list(enc_channels[:-1])[::-1] + [0]  # deepest skip first; last stage has none
        in_chs = [enc_channels[-1]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList(
            DecoderBlock(i, s, o) for i, s, o in zip(in_chs, skip_chs, decoder_channels)
        )

    def forward(self, feats):
        x = feats[-1]
        skips = list(feats[:-1])[::-1]
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
        return x


class FuseBlock(nn.Module):
    """Combine pre and post features at one scale: [pre, post, post - pre] -> C channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = ConvBNReLU(3 * channels, channels)

    def forward(self, pre, post):
        return self.proj(torch.cat([pre, post, post - pre], dim=1))


class SiameseUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        """encoder_weights=None gives a randomly initialized encoder (no download)."""
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        enc_channels = tuple(self.encoder.out_channels[1:])  # drop the identity (input) level

        self.fuse = nn.ModuleList(FuseBlock(c) for c in enc_channels)
        self.loc_decoder = UNetDecoder(enc_channels, decoder_channels)
        self.dmg_decoder = UNetDecoder(enc_channels, decoder_channels)
        self.loc_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        self.dmg_head = nn.Conv2d(decoder_channels[-1], N_DAMAGE_CLASSES, kernel_size=1)

    def forward(self, pre, post):
        if pre.shape != post.shape:
            raise ValueError(f"pre and post must have the same shape, got {tuple(pre.shape)} and {tuple(post.shape)}")
        h, w = pre.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"height and width must be divisible by 32, got {h}x{w}")

        # One pass through the shared encoder for both images (BatchNorm sees both)
        b = pre.shape[0]
        feats = self.encoder(torch.cat([pre, post], dim=0))[1:]
        pre_f = [f[:b] for f in feats]
        post_f = [f[b:] for f in feats]

        loc = self.loc_head(self.loc_decoder(pre_f))
        fused = [fuse(a, c) for fuse, a, c in zip(self.fuse, pre_f, post_f)]
        dmg = self.dmg_head(self.dmg_decoder(fused))
        return loc, dmg

    @torch.no_grad()
    def predict(self, pre, post, loc_threshold: float = 0.5):
        """Hard predictions for scoring: (loc, dmg), both (B, H, W).

        loc is a bool building mask; dmg is 0 outside predicted buildings
        and 1..4 (no-damage .. destroyed) inside them. Switches to eval mode.
        """
        self.eval()
        loc_logits, dmg_logits = self(pre, post)
        loc = torch.sigmoid(loc_logits[:, 0]) > loc_threshold
        dmg = (dmg_logits.argmax(dim=1) + 1) * loc
        return loc, dmg
