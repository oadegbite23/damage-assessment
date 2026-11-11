"""
Model architecture for building localization + damage classification.

Approach: Siamese U-Net — two U-Net encoders (often weight-shared)
process the pre- and post-disaster images separately, with features
fused before/during decoding to jointly predict:
    1. Building localization mask (binary: building / no building)
    2. Damage classification mask (4-class, only scored on building
       pixels): no-damage, minor-damage, major-damage, destroyed

This mirrors common approaches from the xView2 challenge leaderboard
(e.g., "Siamese" and "Fused" architectures — see the xView2 Challenge
paper by Gupta et al., 2019).

TODO:
- [ ] Define Siamese U-Net (via segmentation-models-pytorch as a
      starting encoder/decoder backbone)
- [ ] Define combined loss (localization loss + damage classification
      loss, weighted per xView2 challenge metric conventions)
- [ ] Define optimizer + LR schedule
"""
