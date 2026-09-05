# Catan RL Agent

A self-play reinforcement learning agent for Settlers of Catan, built on the
[Catanatron](https://github.com/bcollazo/catanatron) rules engine, plus a
full evaluation toolkit, a critic-priced trading layer, and a browser UI to
play against (or alongside) the trained bots — solo or with friends.

## What's here

- **PPO self-play training** with a masked discrete action head, potential-based
  reward shaping, and a checkpoint pool for stable multi-agent self-play.
- **Two board encoders**, compared head-to-head: a flat MLP (the one that
  actually wins) and a relational message-passing encoder (a documented
  negative result — see below).
- **An entity-addressed ("pointer") action head** and a **distillation
  pipeline** to warm-start a new architecture from an existing checkpoint
  without training from scratch.
- **Player-to-player trading**, bolted onto an engine that doesn't natively
  support it, priced by the agent's own trained value function — with a
  calibration step and a relative-gain acceptance rule that turned out to
  matter a lot (see Results).
- **Evaluation tools**: head-to-head win rates with confidence intervals,
  an Elo arena across every checkpoint you've trained, and a placement-quality
  probe that measures a very specific skill (opening settlement choice)
  in isolation.
- **A browser UI** (`play.py`) to play solo against the bot, trade with it,
  and watch dev cards and the robber rendered properly — plus a **multiplayer
  lobby** (`lobby.py`) so friends can join from their own devices.

## Repo layout

```
catan_rl/
  streams.py           # splits catanatron's feature vector into board/private/
                       # opponent-set/global streams for the network
  env.py               # env wrapper: action masking, reward shaping,
                       # derived production features, obs normalization
  model.py             # flat-MLP actor-critic (the baseline that won)
  relational.py        # message-passing board encoder (see Results: lost)
  pointer_model.py      # entity-addressed action head over relational
                       # embeddings (fixes the pooling bottleneck relational hit)
  ppo.py               # GAE, clipped surrogate, value clipping, entropy bonus
  opponents.py         # NNPlayer (checkpoint as an in-engine bot) + trading hooks
  trading.py           # player-to-player trading: offer vocabulary, critic
                       # pricing, calibration-aware relative-gain acceptance
  calibrate.py         # fits critic value -> win-probability mapping
  distill.py           # behavior-clone a teacher checkpoint into a new
                       # architecture as a PPO warm start
  llm_negotiator.py    # optional local-LLM negotiation layer (inert unless
                       # a local LLM is actually running; never decides
                       # anything, only phrases/parses)
  train.py             # training entry point + curriculum + CLI
  evaluate.py          # win rate vs heuristics or another checkpoint,
                       # with error bars and trading modes
  arena.py             # round-robin tournament -> Elo across checkpoints
  probe_placement.py   # measures initial-settlement quality specifically
  watch.py             # generates a standalone HTML game replay
  play.py              # browser game: solo vs bots, trading UI, dev cards
  lobby.py             # multiplayer: room setup, shareable join link
  plot_training.py     # training-curve plots from metrics.csv
setup_cse.sh, run_train.sh  # convenience scripts for a shared Linux box
```

## Setup

Needs Python 3.9+ (torch and gymnasium won't install on much older Pythons —
if the system Python is ancient, install a fresh one via
[Miniforge](https://github.com/conda-forge/miniforge) rather than fighting it).

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install catanatron==3.2.1 catanatron_gym==4.0.0 gymnasium==0.29.1 numpy matplotlib
```

Sanity check before committing to a long run:

```bash
python -m catan_rl.train --smoke --derived --prod-potential 0.4
```

## Training

The recipe that won every comparison run in this project — flat encoder,
production-shaped placement rewards, a real warmup phase:

```bash
nohup nice -n 10 python -m catan_rl.train --iters 8000 \
  --encoder flat --derived --prod-potential 0.4 \
  --warmup-iters 200 --pool-refresh 50 --save-every 200 \
  --seed 0 --out checkpoints/gen3 > logs/gen3.log 2>&1 &
```

`--resume` continues an interrupted run from its last checkpoint with the
same command. Key flags: `--encoder {flat,relational,pointer}`, `--derived`
(engineered production features), `--prod-potential` (potential-based
placement-quality shaping — this is what actually fixed the "bad opening
placement" problem), `--warmup-iters` (heuristic-bot phase before self-play).

## Evaluating

```bash
# win rate with confidence interval, optionally with trading
PYTHONHASHSEED=0 python -m catan_rl.evaluate --ckpt A.pt --vs-ckpt B.pt \
    --games 200 --trade agent      # off | agent | table

# Elo across every checkpoint you care about, in one round-robin
PYTHONHASHSEED=0 python -m catan_rl.arena --ckpts A.pt B.pt C.pt \
    --baselines weighted random --games 80

# how good are its opening settlement choices, specifically
python -m catan_rl.probe_placement --ckpt A.pt --games 50
```

`PYTHONHASHSEED=0` matters: heuristic opponents iterate hash-ordered
collections, so without it identical commands produce different games.

## Trading

The pinned engine has no player-to-player trade action at all, so it's
implemented as a wrapper (`trading.py`): a proposer enumerates affordable
1:1/2:1 swaps, prices them with its own value head in one batched forward
pass, and any other network player accepts if its own critic agrees. No
retraining needed — the critic already prices resources in context.

```bash
python -m catan_rl.calibrate --ckpt A.pt --games 60   # V -> win-probability map
python -m catan_rl.evaluate --ckpt A.pt --vs-ckpt B.pt --games 200 --trade agent
```

`--trade agent` (only the evaluated agent proposes) is the mode that can
actually show an effect; `--trade table` (everyone proposes) is symmetric and
cannot move relative standing by construction — a mistake worth not
repeating, since it cost a full experiment cycle here.

## Playing it yourself

```bash
python -m catan_rl.play --ckpt checkpoints/gen3/ckpt_08000.pt --port 8377 --trade
```

Then tunnel `8377` to your laptop (`ssh -L 8377:localhost:8377 <user>@<host>`)
and open `http://localhost:8377`. You're BLUE; the checkpoint plays the other
three seats. Board moves are clickable highlights, dev cards render as actual
cards (locked/dimmed the turn you buy them), and bots will offer you trades
you can accept or reject, plus a builder to send your own.

For friends on other devices:

```bash
python -m catan_rl.lobby --ckpt checkpoints/gen3/ckpt_08000.pt --port 8377 --share
```

`--share` opens a public tunnel (cloudflared, or ssh+localhost.run as a
fallback) and prints a link anyone can open directly — no SSH access needed
on their end.

## Results (why the recipe above looks the way it does)

- **Flat MLP beat a relational message-passing encoder**, 13.5% vs 25% fair
  share at matched training steps — the opposite of what the architecture
  predicted. Diagnosis: the relational encoder pools all node embeddings into
  one board summary *before* the policy head scores node-specific actions
  ("build here"), which destroys the per-node resolution those decisions
  need. `pointer_model.py` is the fix (score each action from its own
  entity's embedding), warm-started via `distill.py` rather than trained
  from scratch.
- **Production-potential shaping fixed weak opening placement.** The original
  reward was pure-VP potential, and both opening settlements are worth
  exactly 1 VP regardless of quality — so placement quality got zero direct
  shaping signal. Adding expected production to the potential function
  (`--prod-potential`) raised placement quality (measured via
  `probe_placement.py`) from ~80% to ~85% choice-percentile, reproducibly
  across independent runs.
- **Trading only helps if it's priced well — badly, it's actively harmful.**
  An early version let both sides "gain" from every trade, which is
  impossible in a zero-sum race and signals miscalibration. Fixing it took
  three changes: calibrating the critic to win-probability units so the two
  sides' gains are comparable, a *relative*-gain acceptance rule (mine must
  exceed theirs, not just be positive), and evaluating with only one side
  trading (`--trade agent`) instead of granting it to everyone symmetrically
  (which cannot show an effect by construction). Net result: 60.5% → 87.5%
  win rate against a fixed opponent. The control that proves it's the pricing
  and not just "having the initiative": letting the same agent propose
  *random* legal offers instead of critic-selected ones collapsed its win
  rate to 3% — worse than not trading at all.

## Known limitations / good next steps

- gen3's specific numbers rest on a single training seed; the earlier
  (superseded) recipe was validated across three seeds.
- Trading is currently a decision-time heuristic layered on a policy that
  never saw trading during training. Training with trade-proposal actions
  in the masked head (rather than as an external wrapper) is the natural
  next step.
- The pinned engine (Catanatron 3.2.1) doesn't enforce "can't play a
  development card the turn you bought it" — `play.py` enforces this as a
  house rule for the human player only, since the trained checkpoints never
  saw that restriction and holding them to it now would be a train/inference
  mismatch.
- No human-evaluation numbers yet — the lobby is built and tested, but a real
  game night hasn't been run.
#   c a t a n _ b o t  
 