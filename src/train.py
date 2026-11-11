"""
Training loop for the building localization + damage classification model.

TODO:
- [ ] Implement training loop with validation checks
- [ ] Log metrics per epoch (localization F1, damage classification F1)
- [ ] Save best checkpoint (gitignored — see .gitignore)
- [ ] Handle class imbalance (damage classes are typically imbalanced —
      "no damage" and "destroyed" tend to dominate depending on the
      disaster type; consider weighted loss or focal loss)
"""
