#!/usr/bin/env bash
# The shipped gsplat training settings, in one place so gsplat_train.sh and the
# sweep's baseline configuration cannot drift apart. Source it; don't execute it.

# Densification settings shared by every DefaultStrategy run.
GSPLAT_STRATEGY_OPTIONS="--strategy.prune-opa 0.002 --strategy.absgrad \
--strategy.grow-scale3d 0.01 --strategy.prune-scale3d 0.1 \
--strategy.prune-scale2d 0.15 --strategy.refine-start-iter 500 \
--strategy.refine-stop-iter 20000 --strategy.refine-every 100 \
--strategy.reset-every 3000"

# Settings the sweep varies. --opacity_reg/--scale_reg penalize low-opacity and
# oversized Gaussians (floaters); gsplat's DefaultStrategy defaults both to 0.0.
GSPLAT_GROW_GRAD2D=0.0006
GSPLAT_FLOATER_REG_WEIGHT=0.01
