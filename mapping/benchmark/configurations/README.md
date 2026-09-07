# Sweep configurations

Each file is layered over `gsplat_server/config/gsplat_train_defaults.proto.txt`
and names only what it changes. The defaults are the MCMC configuration the
sweep selected, so every DefaultStrategy run re-declares `strategy`,
`ssim_lambda` and `packed` to undo them, and 01-07 pin `prune_opa: 0.002` to
match the sweep they are compared against.

See `doc/gsplat_parameter_sweep.md` for the measured results.


# Result PR#21

## Metrics and point-cloud geometry

Reconstruction quality on the held-out views, then what the saved point cloud
actually contains. Best value per column in bold.

| Run | Gaussians | PSNR | SSIM | LPIPS | Median opacity | Faint <0.1 | Median radius | Radius / spacing | Train time | Artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `00_baseline` | 658,320 | 20.84 | 0.7181 | 0.4790 | 0.008 | 76% | 0.0061 | 0.71 | 12 min | [ply 148 MB](../data/panorama/sweep_windows/00_baseline/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/00_baseline/train.log) |
| `01_no_floater_reg` | 3,826 | 14.27 | 0.5761 | 0.7450 | 1.000 | 45% | 0.0014 | 0.05 | 7 min | [ply 1 MB](../data/panorama/sweep_windows/01_no_floater_reg/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/01_no_floater_reg/train.log) |
| `02_grow_grad2d_0003` | 2,185 | 13.24 | 0.5286 | 0.7640 | **1.000** | **9%** | 0.0005 | 0.04 | 8 min | [ply 0 MB](../data/panorama/sweep_windows/02_grow_grad2d_0003/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/02_grow_grad2d_0003/train.log) |
| `03_grow_grad2d_0002` | 6,811 | 13.83 | 0.5611 | 0.7560 | **1.000** | 16% | 0.0005 | 0.08 | 8 min | [ply 2 MB](../data/panorama/sweep_windows/03_grow_grad2d_0002/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/03_grow_grad2d_0002/train.log) |
| `04_refine_longer` | 810,720 | 15.04 | 0.5967 | 0.7160 | **1.000** | 43% | **0.0068** | 0.58 | 10 min | [ply 182 MB](../data/panorama/sweep_windows/04_refine_longer/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/04_refine_longer/train.log) |
| `05_ssim_050` | 8,013 | 14.08 | 0.5758 | 0.7470 | **1.000** | 36% | 0.0004 | 0.04 | 8 min | [ply 2 MB](../data/panorama/sweep_windows/05_ssim_050/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/05_ssim_050/train.log) |
| `06_no_pose_opt` | 769 | 13.10 | 0.5165 | 0.7690 | **1.000** | 21% | 0.0001 | 0.00 | 7 min | [ply 0 MB](../data/panorama/sweep_windows/06_no_pose_opt/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/06_no_pose_opt/train.log) |
| `07_sh_degree_2` | 2,613 | 13.29 | 0.5172 | 0.7650 | **1.000** | 10% | 0.0004 | 0.14 | 8 min | [ply 0 MB](../data/panorama/sweep_windows/07_sh_degree_2/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/07_sh_degree_2/train.log) |
| `08_mcmc_cap_500k` | 500,000 | 22.35 | 0.7581 | 0.4050 | 0.064 | 59% | 0.0061 | 0.79 | 12 min | [ply 113 MB](../data/panorama/sweep_windows/08_mcmc_cap_500k/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/08_mcmc_cap_500k/train.log) |
| `09_mcmc_cap_900k` | 900,000 | 22.30 | 0.7608 | 0.3960 | 0.035 | 69% | 0.0052 | **0.81** | 16 min | [ply 203 MB](../data/panorama/sweep_windows/09_mcmc_cap_900k/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/09_mcmc_cap_900k/train.log) |
| `10_mcmc_cap_1500k` | 1,500,000 | 22.33 | 0.7623 | 0.3910 | 0.015 | 77% | 0.0046 | 0.79 | 20 min | [ply 338 MB](../data/panorama/sweep_windows/10_mcmc_cap_1500k/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/10_mcmc_cap_1500k/train.log) |
| `11_mcmc_cap_900k_ssim050` | 900,000 | **22.45** | **0.7657** | **0.3840** | 0.068 | 58% | 0.0050 | 0.79 | 15 min | [ply 203 MB](../data/panorama/sweep_windows/11_mcmc_cap_900k_ssim050/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/11_mcmc_cap_900k_ssim050/train.log) |
| `12_mcmc_cap_2000k_sh2` | **2,000,000** | 22.12 | 0.7587 | 0.3950 | 0.004 | 82% | 0.0043 | 0.78 | 20 min | [ply 290 MB](../data/panorama/sweep_windows/12_mcmc_cap_2000k_sh2/ply/point_cloud_29999.ply) · [log](../data/panorama/sweep_windows/12_mcmc_cap_2000k_sh2/train.log) |

- **Median opacity** — Median of sigmoid(opacity). Near zero means the Gaussians barely contribute.
- **Faint <0.1** — Fraction of Gaussians with opacity below 0.1
- **Median radius** — Median of mean(exp(scale)) in world units
- **Radius / spacing** — Median radius divided by estimated median nearest-neighbour distance. Below ~0.5 the Gaussians are too small to tile a surface, leaving gaps.


## Settings under test

Only the config values that differ between runs.

| Parameter | `00_baseline` | `01_no_floater_reg` | `02_grow_grad2d_0003` | `03_grow_grad2d_0002` | `04_refine_longer` | `05_ssim_050` | `06_no_pose_opt` | `07_sh_degree_2` | `08_mcmc_cap_500k` | `09_mcmc_cap_900k` | `10_mcmc_cap_1500k` | `11_mcmc_cap_900k_ssim050` | `12_mcmc_cap_2000k_sh2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cap_max` | 900000 | 900000 | 900000 | 900000 | 900000 | 900000 | 900000 | 900000 | 500000 | 900000 | 1500000 | 900000 | 2000000 |
| `floater_reg_weight` | 0.01 | None | None | None | None | None | None | None | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `grow_grad2d` | 0.0006 | 0.0006 | 0.0003 | 0.0002 | 0.0003 | 0.0003 | 0.0003 | 0.0002 | 0.0006 | 0.0006 | 0.0006 | 0.0006 | 0.0006 |
| `packed` | None | None | None | True | True | None | None | True | None | True | True | True | True |
| `pose_opt` | True | True | True | True | True | True | None | True | True | True | True | True | True |
| `prune_opa` | 0.005 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| `refine_stop_iter` | 20000 | 20000 | 20000 | 20000 | 25000 | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 |
| `reset_every` | 3000 | 3000 | 3000 | 3000 | 5000 | 3000 | 3000 | 3000 | 3000 | 3000 | 3000 | 3000 | 3000 |
| `sh_degree` | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 2 |
| `ssim_lambda` | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.5 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.5 | 0.2 |
| `strategy` | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_DEFAULT | STRATEGY_MCMC | STRATEGY_MCMC | STRATEGY_MCMC | STRATEGY_MCMC | STRATEGY_MCMC |
