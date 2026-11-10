# SUNG-CQL CrowdNav (V5 attention, fixed 65-D input)

The user requires exactly 65 original CrowdNav state features and 2 world-frame
actions. Dataset observations and the original IQL state transform are preserved.
No goal-direction features are added or substituted. The V3 67-D experiment was
stopped and is not a valid comparison under this interface. Its files are retained
only as historical output; V4 rejects its checkpoints.

## Training

- Offline: 100000 minibatch update rounds, including 10000 BC warmup rounds.
  Each round also updates the VAE; there is no additional VAE pretraining stage.
- Online: 50000 actual update rounds, by default one update after each episode.
  Environment steps are logged separately. Evaluation and saving use update counts.
- Standalone --mode vae uses --vae_steps independently.
- VAE latent size is 2*(65+2)=134; this internal latent space is not policy input.
- Fixed offline statistics normalize actor/critic inputs consistently with inference.
- Training validation uses val; final offline performance uses 500 test cases.
  offline_final_result.json records whether the requested 0.60 target was reached.

## V4 tuning experiment

The full V2 model obtained 28/500 test successes (5.6%). Validation was 18% at the
end of warmup and dropped during CQL. The coordinate diagnosis did not establish
that missing direction caused this performance, and the user rejected expanding
the inputs. V4 keeps the exact original representation and changes training knobs:

| Parameter | V2 | V4 |
| --- | --- | --- |
| CQL minimum-Q weight | 5 | 1 |
| SAC target entropy | -2 | -4 |
| Initial entropy temperature | 1 | 0.1 |
| BC warmup entropy weight | 1 | 0 |

These are navigation tuning choices, not paper-default settings or verified fixes.
Lower regularization and lower entropy are being tested after the previous run's
policy degradation. Zero BC entropy weight makes warmup likelihood-only; CQL
actor/critic and SUNG uncertainty-guided online losses otherwise remain intact.
All four parameters are exposed on the CLI. Success improvement remains unverified.

Reference: supplied sung.pdf, Algorithm 1, Section V-A and Appendix C;
https://github.com/guosyjlu/SUNG/tree/main/CQL . The upstream replay_p convention
is the global-buffer fraction. Empty timeout annotations preserve dataset terminal
flags; online timeouts bootstrap unless --timeout_terminal is set.

## Run and outputs

Run from repository root:

```powershell
python -m crowd_nav.sung_cql --mode offline --offline_updates 100000 --eval_freq 5000
python -m crowd_nav.sung_cql --mode online --online_updates 50000 --eval_freq 5000
```

Default output: crowd_nav/data/output_sung_cql_v5_attention.
Final and best checkpoints are separate; final evaluation uses the final checkpoint.
Online starts a new run from a ready offline checkpoint. Replay/environment resume
from online checkpoints is not implemented. Old checkpoints/configurations must
not be silently mixed with the new entropy and regularization settings.

## Checks

python -m unittest crowd_nav.tests.test_sung_input_contract verifies the 65/2
interface, rejects 67-D input, and compares live input exactly to the original
transform. python -m crowd_nav.tests.smoke_test checks training updates, VAE,
checkpoints, timeout handling, evaluation RNG isolation and exact update budgets.


## V5 investigation and change

V4 completed 100000 updates: final test success 91/500 (18.2%), 395 collisions
and 14 timeouts. Its best logged validation success was 28%, with final validation
10%. Reducing CQL/entropy weights alone did not meet the requested 60% target.

The offline CQL objective was compared with the young-geng/SimpleSAC reference;
no new discrepancy in its default non-Lagrangian loss was established. However,
this repository's IQL uses SelfAttention and its policy/cql.py uses ST human
encoding, whereas the SUNG adapter had omitted human encoding entirely. On 10000
stored states, reversing only human order changed V4 deterministic action L2 by
0.048/0.118/0.260 at the 50th/90th/99th percentiles and changed Q1 by 0.873 on
average. These measurements expose order sensitivity, not proof of its sole
responsibility for low success.

V5 adds a CrowdNav-style attention encoder to the actor and both critics:
65 external features -> five 13-feature human rows -> shared 150/100 and 100/50
MLPs -> global-context softmax attention -> 6 self features + 50 pooled features.
The learned 56-feature representation is INTERNAL; states, replay and environment
interfaces remain exactly 65-D and actions exactly 2-D. Targets use identical
encoders. --encoder mlp retains the previous architecture for controlled ablation.
All V4 training hyperparameters are retained to isolate the encoding change.

Statistics are shared across the five human slots, computed with float64
accumulation and stored as float32. Per-slot normalization would otherwise break
permutation invariance even with attention. Attention uses a stable softmax,
without the legacy encoder's zero-score mask or GPU-to-CPU logging side effect.
The CQL candidate-action dimensions and policy likelihood paths use the encoder
consistently for arbitrary batch size and candidate count. VAE remains the paper's
state-action reconstruction model; its uncertainty is not claimed permutation
invariant. No CQL loss, reward, data filtering, external state feature or safety
controller is added by V5.

Checks: test_sung_attention verifies identical policy/Q for permutations of a
physical scene, repeated-action likelihood consistency and actor/critic gradients.
The existing input-contract and training smoke tests pass with attention enabled.
V5 checkpoints reject older formats because learned encoder/statistics changed.
Full training is started in a new output directory; success is not yet verified.


## Online run and plotting logs

Online loads the FINAL V5 offline checkpoint (100000 updates). It runs 50000
updates, one per completed episode; evaluates 50 fixed test cases initially and
every 5000 updates, and saves every 5000 updates. Output is in
crowd_nav/data/output_sung_cql_v5_attention/online/.

online.log uses timestamp - [TEST] summaries and [ONLINE_SUNG] ep=N/50000 lines.
reward and Avg Reward are discounted returns using the configured discount.
nav_time is time in seconds for successful episodes and 0 for collision/timeout;
test Navtime is the mean over successful episodes, or 0 if none succeed.
buffer_size is the current ONLINE ring buffer size, capped at 20000, not total
collected transitions. metrics.jsonl additionally retains undiscounted_reward,
env_steps, update, global_buffer_size, episode counts and test phases for plotting.
The human-readable tag identifies SUNG rather than mislabeling it AWAC.


## Seven-human run (2026-09-15)

By user request, defaults now use Crowd_nav_7/Crowd_nav_7.hdf5 and 91 original
features (7*13), with 2 actions. The human encoder and shared normalization infer
human count from input dimension. The internal pooled feature remains 56; VAE
latent dimension follows the unchanged formula 2*(91+2)=186. --state_dim 65
retains five-human model compatibility when the matching dataset/env are supplied.

All V5 algorithm hyperparameters are unchanged. run_sung_seven.py runs offline
100000 updates (validation every 5000, final 500-case test), then loads its final
checkpoint for online 50000 episode updates (50-case test initially and every
5000). Online metrics and log formatting are unchanged. The launcher snapshots
the provided seven-human env.config and stops on stage failure.

Output: crowd_nav/data/output_sung_cql_v5_attention_7/; online logs/models in its
online/ subdirectory. pipeline_status.json reports stage, process and completion.
Five-human outputs are preserved. Attention/input regression tests and training
smoke tests passed; actual seven-human HDF5 and live environment also passed shape,
finite offline/online loss, normalization and VAE-size checks.
