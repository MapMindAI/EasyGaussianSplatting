#!/usr/bin/env bash
# The gsplat training settings this repo ships, in one place so
# scripts/gsplat_train.sh and the sweep's 00_baseline configuration cannot
# drift apart. Source it; don't execute it.
#
# How to change these:
#   - Edit a value here and both the trainer and the sweep baseline pick it up.
#   - To compare a change against what ships rather than replacing it, add a
#     configuration to mapping/benchmark/gsplat_sweep.sh instead. Flags listed
#     there come after these, and for a repeated flag the last value wins.
#   - "gsplat default" below means DefaultStrategy's own default, for reference.
#
# The main lever on detail is how many Gaussians the run ends with. Lowering
# --strategy.grow-grad2d or --strategy.prune-opa, or raising
# --strategy.refine-stop-iter, all grow that count. DefaultStrategy has no
# upper bound on it, so on a small GPU it grows until training runs out of
# memory; use the mcmc strategy's --strategy.cap-max when it must be bounded.

# Densification settings shared by every DefaultStrategy run.
GSPLAT_STRATEGY_OPTIONS=(
  # Prune Gaussians whose opacity falls below this. gsplat default 0.005,
  # lowered here to keep faint detail the default would discard.
  --strategy.prune-opa 0.002

  # Densify on absolute image-plane gradients (AbsGS) instead of averaged ones.
  # Off in gsplat by default. Resolves detail better, but it raises the
  # gradient scale, so GSPLAT_GROW_GRAD2D below has to rise with it.
  --strategy.absgrad

  # Below this 3D scale (as a fraction of scene size) a densifying Gaussian is
  # duplicated; above it, split instead. gsplat default, unchanged.
  --strategy.grow-scale3d 0.01

  # Prune Gaussians bigger than this fraction of the scene. gsplat default,
  # unchanged. Raise it if large flat surfaces are being eaten away.
  --strategy.prune-scale3d 0.1

  # Prune Gaussians covering more than this fraction of the image. Inert as
  # configured: gsplat applies it only while step < refine_scale2d_stop_iter,
  # which defaults to 0 and is not set here. Set that flag to switch it on.
  --strategy.prune-scale2d 0.15

  # First step that densifies, last one, and how often in between. Start and
  # interval are gsplat defaults; the stop is raised from 15000 so
  # densification keeps working through more of a 30000-step schedule.
  --strategy.refine-start-iter 500
  --strategy.refine-stop-iter 20000
  --strategy.refine-every 100

  # Periodically drop every opacity so unused Gaussians get pruned instead of
  # lingering as haze. gsplat default, unchanged. Raising it keeps more
  # Gaussians alive; lowering it culls harder.
  --strategy.reset-every 3000
)

# Image-plane gradient above which a Gaussian is split or duplicated -- the
# single biggest influence on the final Gaussian count. gsplat's default of
# 0.0002 assumes --strategy.absgrad is off; AbsGS suggests around 0.0008 with
# it on. Lower it for more Gaussians and finer detail, at the cost of memory.
GSPLAT_GROW_GRAD2D=0.0006

# Weight for --opacity_reg and --scale_reg, which penalize low-opacity and
# oversized Gaussians. gsplat's DefaultStrategy defaults both to 0.0; 0.01 is
# its *mcmc* preset. Here it is what keeps the Gaussian count from running
# away, and that is not free: on data/panorama it left 66% of Gaussians below
# opacity 0.1 and a median radius of 0.57x the distance to the nearest
# neighbour, so surfaces were under-covered and detail was lost. Lower it for
# more detail, but watch peak GPU memory -- at 0.0 this dataset reached 1.6 M
# Gaussians by step 9000 and exhausted a 4 GB card.
GSPLAT_FLOATER_REG_WEIGHT=0.01
