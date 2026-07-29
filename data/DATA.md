# Data

The repository includes serialized training and inference trajectories for the
two implemented benchmarks. Every trajectory array uses the layout
`(time, protocol, component)`.

## Drucker–Prager

`data/Drucker_Prager/training_data.pkl` stores a list with:

| Index | Field | Shape or value |
| --- | --- | --- |
| 0 | strain at \(t_k\) | `(250, 18, 2)`, `float32` |
| 1 | strain at \(t_{k+1}\) | `(250, 18, 2)`, `float32` |
| 2 | stress at \(t_k\) | `(250, 18, 2)`, `float32` |
| 3 | stress at \(t_{k+1}\) | `(250, 18, 2)`, `float32` |
| 4 | number of snapshots | scalar |
| 5 | number of protocols | `18` |
| 6 | reduced invariant dimension | `2` |
| 7 | valid stop index per protocol | `(18,)`, `int64` |

The two components are the reduced volumetric/deviatoric invariant
representation used by the benchmark.

`data/Drucker_Prager/inference_data.pkl` has the same first seven fields but no
stop-index array. Its trajectory arrays have shape `(600, 3, 2)` and use
`float64`.

## Isotropic hardening

The training and inference files store:

```text
[strain, strain_next, stress, stress_next,
 n_snapshots, n_protocols, dimension]
```

Training arrays have shape `(260, 2, 1)`; inference arrays have shape
`(615, 1, 1)`. Both use `float32`.
