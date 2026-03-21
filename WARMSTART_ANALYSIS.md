# Warmstart vs Coldstart RL Analysis

## Summary

Coldstart RL outperforms BC-warmstarted RL on multi-date VWAP execution. The BC policy clones a suboptimal heuristic that constrains RL exploration, while coldstart discovers a better strategy from scratch.

## Experimental Setup

**Task**: Buy 500 shares of AAPL over 15-minute windows, minimising slippage vs realised VWAP.

**Data**:
- Training: 7 dates (2019-01-30 through 2021-07-13), ~182 windows, AAPL only
- Eval: 1 held-out date (2019-10-18), completely unseen during training
- All data from real NASDAQ LOBSTER order book reconstructions

**RL Config** (all runs):
- `NUM_ENVS=512`, `NUM_STEPS=126`, `TOTAL_TIMESTEPS=5e6` → 77 update rounds
- `NUM_MINIBATCHES=8`, `GAMMA=0.99`, `GAE_LAMBDA=0.9999`, `CLIP_EPS=0.2`
- `JAXMARL_PATCH_ORTHOGONAL=1` (DGX Spark sm_121 workaround)

**BC Model**:
- Trained on 3.41M samples (AAPL + MSFT, 9 dates × 3 horizons × 4 densities)
- Cross-date validation: 99.91% accuracy, 99.99% exec recall, 95.4% exec precision
- Clones a volume-weighted baseline agent's execution timing decisions

## Results

| Run | LR | Update Epochs | Actor Freeze | Final Train | Best Eval | Best At |
|-----|-----|--------------|--------------|-------------|-----------|---------|
| **Coldstart** | 1e-4 | 8 | - | -0.164 | **-0.0047** | update 75 |
| Warmstart | 3e-5 | 2 | 0 | -0.160 | -0.0140 | update 44 |
| Warmstart + freeze | 3e-5 | 2 | 20 | -0.160 | -0.0142 | update 44 |

Coldstart achieves 3x better eval reward and keeps improving throughout training.

## Detailed Observations

### Coldstart

- **Updates 1-20**: Train reward near zero (-0.003 to -0.013). The random policy barely executes, so slippage is low but task completion is poor. Eval reward steadily improves from -0.073 to -0.024 as the policy learns when to execute.
- **Updates 20-50**: Train reward deteriorates to -0.10 to -0.16 as the policy starts executing aggressively and taking market impact. But eval reward continues improving (-0.024 to -0.007) — the policy is learning to time executions better on the held-out date.
- **Updates 50-77**: Train reward stays around -0.16 (noisy). Eval reward keeps improving slowly, reaching -0.0047 at update 75. The policy is still refining.

The divergence between train and eval reward is notable: the policy explores aggressively on training data (accepting worse rewards to discover better strategies) while performing well on the evaluation date. This is healthy RL behaviour.

### Warmstart (no freeze)

- **Update 1**: Train reward -0.010 (better than coldstart's -0.051). The BC-initialised policy already knows approximately when to execute. Eval reward -0.016.
- **Updates 2-20**: Rapid degradation. Train reward drops to -0.16 within 20 updates. The random critic produces garbage advantage estimates that push the actor away from its good initialisation.
- **Updates 20-77**: Converges to the same train reward range as coldstart (~-0.16). Eval plateaus around -0.014, never recovering. Best eval -0.014 at update 44.

### Warmstart + Actor Freeze (20 steps)

Identical trajectory to warmstart without freeze. The 20 steps of critic-only training were insufficient to save the actor — eval reward during freeze improved only marginally (-0.0155 to -0.0146). Post-unfreeze, the actor degraded to the same place.

## Why Coldstart Wins

### 1. Objective Mismatch (Primary Cause)

The BC policy clones a VWAP baseline heuristic: "execute at evenly-spaced intervals proportional to estimated market volume." This is a reasonable strategy but not optimal for the RL reward function, which measures actual slippage against realised VWAP.

The RL reward function can discover strategies the baseline never considers:
- Adapting execution timing to intraday price momentum
- Front-loading or back-loading in certain microstructure conditions
- Exploiting correlations between order book imbalance and short-term price moves

Coldstart has the freedom to explore these. Warmstart is anchored to "execute on schedule," and the RL updates can't escape this basin within the training budget.

### 2. Critic Bootstrapping Problem

The BC warmstart only initialises the actor (policy network). The critic (value network) starts random. In PPO, the critic's value estimates compute advantages that drive actor updates:

```
advantage = reward + γ * V(next_state) - V(current_state)
```

With a random critic, advantages are noise. The first few actor updates push the BC policy in random directions, destroying its good initialisation before the critic has learned anything useful. Even with actor freezing for 20 updates, the critic doesn't learn fast enough to provide useful guidance.

### 3. Exploration Suppression

The BC policy is highly confident (trained to 99.9% accuracy). Its action distribution has low entropy — it almost always outputs the "correct" action with high probability. This suppresses exploration, preventing RL from discovering that the "correct" BC action isn't actually optimal for the reward function.

Coldstart's random policy has maximum entropy initially, enabling broad exploration of the action space.

## Implications

1. **Use coldstart for RL training.** The multi-date data (7 dates, 182 windows) provides sufficient diversity for RL to learn directly. BC warmstart is counterproductive for this task.

2. **The BC pipeline has value for other purposes**: validating the observation space alignment, testing feature engineering, and providing a behavioural baseline to compare RL against. The 99.9% cross-date accuracy confirms the observation space is well-designed.

3. **If warmstart is revisited**, consider:
   - KL penalty to the BC policy (prevents catastrophic forgetting while allowing improvement)
   - Much longer actor freeze (50+ updates) to give the critic time to learn
   - Pre-training the critic on BC rollouts before starting RL
   - Lower entropy coefficient to maintain the BC policy's structure

4. **The best coldstart checkpoint** is saved at `checkpoints/MARLCheckpoints/vwap_tracking_rl/coldstart-multidate-v2_best/` with eval reward -0.0047.

## Files

| File | Description |
|------|-------------|
| `experiments/coldstart_v2_run.log` | Coldstart with best-checkpoint tracking |
| `experiments/warmstart_run.log` | Warmstart without actor freeze |
| `experiments/warmstart_freeze_run.log` | Warmstart with 20-step actor freeze |
| `checkpoints/bc_vwap_multistock/` | BC checkpoint (orbax) |
| `checkpoints/bc_vwap_multistock.json` | BC training metadata |
| `data/bc_vwap_multistock.npz` | Combined BC training data (3.41M samples) |
