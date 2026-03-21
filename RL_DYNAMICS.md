# RL Training Dynamics

Reference for how PPO training parameters interact in this codebase.

## The Task

Buy `task_size` shares (default 500) during a `episode_time` window (default 900s = 15 min) using a hold/execute action space (`fixed_quant_value` shares per execution, default 10). The reward tracks how close the agent's average execution price is to VWAP.

Each environment step processes `n_data_msg_per_step` (default 100) real market messages from historical LOBSTER data, then the agent decides: hold or execute. A 15-minute window produces roughly 60-130 steps depending on the day's activity.

## PPO Training Loop

```
Repeat for NUM_UPDATES rounds:

    1. COLLECT: Run NUM_ENVS environments for NUM_STEPS each
       → NUM_ENVS × NUM_STEPS transitions

    2. LEARN: For each of UPDATE_EPOCHS passes:
         Shuffle all transitions
         Split into NUM_MINIBATCHES chunks
         Do one gradient step per chunk
       → NUM_MINIBATCHES × UPDATE_EPOCHS gradient steps per round

    3. Discard data, repeat
```

## Key Derived Quantities

```
transitions_per_round = NUM_ENVS × NUM_STEPS
minibatch_size        = NUM_ENVS × NUM_STEPS / NUM_MINIBATCHES
NUM_UPDATES           = TOTAL_TIMESTEPS / (NUM_ENVS × NUM_STEPS)
grad_steps_per_round  = NUM_MINIBATCHES × UPDATE_EPOCHS
total_grad_steps      = NUM_UPDATES × NUM_MINIBATCHES × UPDATE_EPOCHS
                      = TOTAL_TIMESTEPS × UPDATE_EPOCHS / (NUM_STEPS × minibatch_size)
```

## Scaling NUM_ENVS and NUM_MINIBATCHES Together

Doubling both `NUM_ENVS` and `NUM_MINIBATCHES` preserves learning dynamics:

| Quantity | Effect |
|----------|--------|
| `minibatch_size` | Unchanged (both numerator and denominator double) |
| `NUM_UPDATES` | Halved (2x envs consumes timesteps 2x faster) |
| `grad_steps_per_round` | Doubled (2x minibatches) |
| `total_grad_steps` | **Unchanged** (halving and doubling cancel) |

The only difference is each rollout samples more of the environment distribution (more diverse windows in parallel), which generally improves gradient quality. Increase both until GPU memory becomes the bottleneck.

`TOTAL_TIMESTEPS` is the training budget — increase it based on whether the reward curve has converged, not based on hardware.

## NUM_STEPS and Episode Length

`NUM_STEPS` is the PPO rollout length, not the episode time horizon. If an episode finishes mid-rollout, the env auto-resets and a new episode begins within the same rollout. If an episode is still running when the rollout ends, the critic's value estimate bootstraps the unknown future reward:

```python
delta = reward + gamma * next_value * (1 - done) - value
```

`NUM_STEPS=126` was chosen to roughly match one full episode (~60-130 steps). If `NUM_STEPS` is much shorter than episode length, most episodes bootstrap rather than completing, and early training relies heavily on an inaccurate critic.

## UPDATE_EPOCHS

Not equivalent to a learning rate multiplier. Each epoch re-shuffles transitions into different minibatch compositions, and PPO's clipping constraint becomes increasingly binding across epochs:

```python
ratio = exp(new_log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
```

Early epochs make large updates (ratio near 1.0, clipping rarely triggers). Later epochs are constrained (ratio has drifted, clipping limits further change). This provides diminishing, stabilizing refinement — unlike a single large-LR step which has no such safety mechanism. Typical range is 3-10.

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_ENVS` | 2048 | Parallel environment copies |
| `NUM_STEPS` | 126 | Steps collected per env per rollout (~1 episode) |
| `NUM_STEPS_EVAL` | 63 | Steps for evaluation rollouts |
| `UPDATE_EPOCHS` | 8 | Passes over collected data per update |
| `NUM_MINIBATCHES` | 8 | Chunks per pass (minibatch_size = transitions / this) |
| `TOTAL_TIMESTEPS` | 2e7 | Total env steps before stopping |
| `LR` | 1e-4 | Learning rate |
| `GAMMA` | 0.99 | Discount factor for future rewards |
| `GAE_LAMBDA` | 0.9999 | Bias-variance tradeoff in advantage estimation |
| `CLIP_EPS` | 0.2 | PPO clipping range [1-eps, 1+eps] |
| `ENT_COEF` | 0.001 | Entropy bonus (encourages exploration) |
| `VF_COEF` | 5e-5 | Value function loss weight |
| `MAX_GRAD_NORM` | 0.5 | Gradient clipping |
| `ACTOR_FREEZE_STEPS` | 0 | Update steps to freeze actor (warmstart only) |

## Checkpointing

The training loop saves checkpoints every update step, keeping the 2 most recent plus one from the halfway point. When `CALC_EVAL` is true, the best model by eval reward is additionally saved to `{checkpoint_dir}_best/` (max 1 kept).
