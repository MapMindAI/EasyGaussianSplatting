# Sweep configurations

Each file is layered over `gsplat_server/config/gsplat_train_defaults.proto.txt`
and names only what it changes. The defaults are the MCMC configuration the
sweep selected, so every DefaultStrategy run re-declares `strategy`,
`ssim_lambda` and `packed` to undo them, and 01-07 pin `prune_opa: 0.002` to
match the sweep they are compared against.

See `doc/gsplat_parameter_sweep.md` for the measured results.
