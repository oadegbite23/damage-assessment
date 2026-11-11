"""
Evaluation utilities.

The xView2 challenge uses a combined F1 metric:
    F_xView2 = 0.3 * F1(localization) + 0.7 * F1(damage classification)

TODO:
- [ ] Localization F1 (building vs. no building, pixel-wise)
- [ ] Damage classification F1 (per damage class, then averaged —
      check whether to use macro or the harmonic-mean weighting used
      in the original xView2 challenge)
- [ ] Combined xView2 metric
- [ ] Per-disaster-type breakdown (useful for understanding
      generalization across disaster types, e.g. hurricane vs.
      earthquake vs. wildfire)
"""
