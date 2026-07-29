# Checkpoints

The example scripts reconstruct their architecture and normalization before
loading a PyTorch state dictionary with `weights_only=True`.

## Canonical checkpoints

| File | Used by | Precision |
| --- | --- | --- |
| `drucker_prager.pt` | `drucker_prager_inference.py` | `float64` |
| `elasto_plastic_hardening.pt` | `isotropic_hardening_inference.py` | `float32` |

The training scripts write `drucker_prager_retrained.pt` and
`isotropic_hardening_retrained.pt`, respectively, so these canonical files are
not overwritten.
