"""Training loop (Section 6).

Curriculum:
  Phase A (warmup): enemies are heuristic bots (Random / WeightedRandom).
  Phase B (self-play): enemies sampled per environment from the opponent pool
  (current frozen checkpoints + heuristics), refreshed every pool_refresh
  iterations.

Usage:
  python -m catan_rl.train --iters 200 --num-envs 8
  python -m catan_rl.train --smoke        # tiny run to validate the pipeline
"""

import argparse
import os
import time

import numpy as np
import torch

# Shared-machine etiquette: this workload is dominated by the pure-Python
# Catan simulator, so extra BLAS threads add contention, not speed.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from .env import DERIVED_SIZE, CatanEnv, RunningNorm, SyncVecCatan
from .model import ActorCritic, clean_state_dict
from .opponents import OpponentPool
from .ppo import PPOConfig, RolloutBuffer, ppo_update
from .streams import build_stream_index

NUM_ACTIONS = 290
NUM_PLAYERS = 4


def make_vec(num_envs, enemy_factories, norm, cfg, seed, prod_potential=0.0,
             derived=False):
    def env_fn():
        return CatanEnv(
            enemy_factories=enemy_factories,
            gamma=cfg.gamma,
            shaping_coef=0.1,
            prod_potential=prod_potential,
            derived=derived,
            normalizer=norm,
        )

    return SyncVecCatan(num_envs, env_fn, base_seed=seed)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sindex = build_stream_index(NUM_PLAYERS, num_derived=DERIVED_SIZE if args.derived else 0)
    cfg = PPOConfig(rollout_steps=args.rollout_steps)

    def model_factory():
        if args.encoder == "pointer":
            from .pointer_model import PointerActorCritic

            return PointerActorCritic(sindex, NUM_ACTIONS)
        return ActorCritic(sindex, NUM_ACTIONS, encoder=args.encoder)

    model = model_factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    probe = CatanEnv(derived=args.derived)
    norm = RunningNorm(probe.obs_size)
    del probe

    start_iter = 1
    if args.resume:
        import glob

        ckpts = sorted(glob.glob(os.path.join(args.out, "ckpt_*.pt")))
        if ckpts:
            data = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(data["model"])
            norm.load_state_dict(data["norm"])
            if "optimizer" in data:
                optimizer.load_state_dict(data["optimizer"])
            start_iter = data["iter"] + 1
            print(f"resumed from {ckpts[-1]} (iter {data['iter']})", flush=True)
        else:
            print("no checkpoint found; starting fresh", flush=True)

    if args.compile:
        model.board_enc = torch.compile(model.board_enc)

    base_lr, base_ent = cfg.lr, cfg.entropy_coef
    final_lr, final_ent = args.final_lr, args.final_entropy

    pool = OpponentPool(model_factory, derived=args.derived)
    vec = make_vec(args.num_envs, None, norm, cfg, args.seed, args.prod_potential, args.derived)  # Phase A
    obs, masks = vec.reset()

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.csv")
    if start_iter == 1 or not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("iter,steps,winrate,entropy,policy_loss,value_loss,clip_frac,elapsed_s\n")
    win_history = []
    t0 = time.time()

    for it in range(start_iter, args.iters + 1):
        frac = min(it / max(args.iters, 1), 1.0)  # linear anneal over the run
        cfg.entropy_coef = base_ent + (final_ent - base_ent) * frac
        lr_now = base_lr + (final_lr - base_lr) * frac
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        buf = RolloutBuffer(cfg.rollout_steps, args.num_envs, obs.shape[1], NUM_ACTIONS)
        for _ in range(cfg.rollout_steps):
            actions, logprobs, values = model.act(obs, masks)
            next_obs, next_masks, rewards, dones, wins = vec.step(actions)
            buf.add(obs, masks, actions, logprobs, rewards, dones, values)
            win_history.extend(wins)
            obs, masks = next_obs, next_masks

        with torch.no_grad():
            _, last_values = model.dist_value(
                torch.as_tensor(obs), torch.as_tensor(masks)
            )
        stats = ppo_update(model, optimizer, buf, last_values.numpy(), cfg)

        recent = win_history[-100:]
        winrate = float(np.mean(recent)) if recent else float("nan")
        print(
            f"iter {it:4d} | steps {it * cfg.rollout_steps * args.num_envs:8d} | "
            f"win% (last {len(recent)} eps) {winrate:5.2f} | "
            f"ent {stats['entropy']:.3f} | pi {stats['policy_loss']:+.4f} | "
            f"v {stats['value_loss']:.4f} | clip {stats['clip_frac']:.2f} | "
            f"{time.time() - t0:6.0f}s",
            flush=True,
        )
        with open(metrics_path, "a") as f:
            f.write(
                f"{it},{it * cfg.rollout_steps * args.num_envs},{winrate:.4f},"
                f"{stats['entropy']:.4f},{stats['policy_loss']:.5f},"
                f"{stats['value_loss']:.5f},{stats['clip_frac']:.4f},"
                f"{time.time() - t0:.0f}\n"
            )

        # Phase B: enter self-play once warmup done; refresh pool periodically
        if it == args.warmup_iters or (
            it > args.warmup_iters and it % args.pool_refresh == 0
        ):
            pool.push(model, norm)
            vec = make_vec(
                args.num_envs, None, norm, cfg, args.seed + it, args.prod_potential,
                args.derived
            )
            # per-env lineup sampling
            for e in vec.envs:
                e.__init__(
                    enemy_factories=pool.sample_enemy_factories(),
                    gamma=cfg.gamma,
                    shaping_coef=0.1,
                    prod_potential=args.prod_potential,
                    derived=args.derived,
                    normalizer=norm,
                )
            obs, masks = vec.reset()
            print(f"  -> refreshed opponent pool ({len(pool.checkpoints)} checkpoints)")

        if it % args.save_every == 0 or it == args.iters:
            path = os.path.join(args.out, f"ckpt_{it:05d}.pt")
            torch.save(
                dict(model=clean_state_dict(model), norm=norm.state_dict(), iter=it,
                     optimizer=optimizer.state_dict(), encoder=args.encoder,
                     derived=args.derived), path
            )
            print(f"  -> saved {path}")

    return model, norm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--warmup-iters", type=int, default=30)
    p.add_argument("--pool-refresh", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--encoder", choices=["flat", "relational", "pointer"], default="flat")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the board encoder (helps relational on CPU)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint in --out")
    p.add_argument("--derived", action="store_true",
                   help="append engineered production features (own per-resource "
                        "pips, totals) to the observation")
    p.add_argument("--prod-potential", type=float, default=0.0,
                   help="weight of expected-production term in the shaping "
                        "potential (0.3-0.5 recommended; gives placement "
                        "quality a direct learning signal)")
    p.add_argument("--final-lr", type=float, default=1e-4)
    p.add_argument("--final-entropy", type=float, default=0.005)
    p.add_argument("--smoke", action="store_true", help="tiny pipeline validation run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.rollout_steps = 3, 2, 64
        args.warmup_iters, args.pool_refresh, args.save_every = 2, 1, 3
    train(args)
