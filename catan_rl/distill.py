"""Distill a teacher checkpoint into a fresh student architecture.

Plays the teacher (all four seats, stochastic for coverage) and records every
non-forced decision from every seat, then behavior-clones the student on
(observation, legal mask, action) with cross-entropy. Output is a normal
checkpoint named ckpt_00000.pt, so PPO fine-tuning is just:

  python -m catan_rl.distill --teacher checkpoints/gen3/ckpt_08000.pt \
      --student-encoder pointer --games 150 --out checkpoints/gen4
  python -m catan_rl.train --encoder pointer --derived --prod-potential 0.4 \
      --resume --out checkpoints/gen4 --iters 8000 ...

The student inherits the teacher's derived-features setting.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from catanatron.game import Game
from catanatron.models.player import Color

from .env import DERIVED_SIZE, RunningNorm
from .model import ActorCritic, clean_state_dict
from .opponents import NNPlayer
from .streams import build_stream_index
from .watch import load_agent

NUM_ACTIONS = 290
SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]


class Recorder(NNPlayer):
    obs_buf, mask_buf, act_buf = [], [], []

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        import numpy as np
        from catanatron_gym.envs.catanatron_env import (
            ACTION_SPACE_SIZE, from_action_space, to_action_space)
        from catanatron_gym.features import create_sample_vector

        obs = np.asarray(create_sample_vector(game, self.color), dtype=np.float64)
        if self.derived:
            from .env import derived_features

            obs = np.concatenate([obs, derived_features(game.state, self.color)])
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[to_action_space(a) for a in playable_actions]] = True

        norm_obs = self.normalizer(obs) if self.normalizer is not None else obs
        with torch.no_grad():
            action, _, _ = self.model.act(
                norm_obs[None].astype(np.float32), mask[None],
                deterministic=self.deterministic)
        a = int(action[0])
        Recorder.obs_buf.append(obs.astype(np.float32))  # RAW obs; student fits its own norm
        Recorder.mask_buf.append(mask)
        Recorder.act_buf.append(a)
        return from_action_space(a, playable_actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student-encoder", choices=["flat", "relational", "pointer"],
                   default="pointer")
    p.add_argument("--games", type=int, default=150)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    teacher, tnorm = load_agent(args.teacher)
    derived = getattr(teacher, "uses_derived", False)
    print(f"teacher: {args.teacher} (derived={derived})")

    Recorder.obs_buf, Recorder.mask_buf, Recorder.act_buf = [], [], []
    for g in range(args.games):
        players = [Recorder(c, teacher, tnorm, deterministic=False, derived=derived)
                   for c in SEAT_COLORS]
        Game(players, seed=args.seed + g).play()
        if (g + 1) % 25 == 0:
            print(f"  {g+1}/{args.games} games, {len(Recorder.act_buf)} decisions")

    obs = torch.as_tensor(np.stack(Recorder.obs_buf))
    masks = torch.as_tensor(np.stack(Recorder.mask_buf))
    acts = torch.as_tensor(np.array(Recorder.act_buf))
    print(f"dataset: {len(acts)} decisions from {args.games} games")

    norm = RunningNorm(obs.shape[1])
    norm.update(obs.numpy().astype(np.float64))
    norm.frozen = True
    obs_n = torch.as_tensor(
        np.stack([norm(o) for o in obs.numpy().astype(np.float64)]))

    sindex = build_stream_index(4, num_derived=DERIVED_SIZE if derived else 0)
    if args.student_encoder == "pointer":
        from .pointer_model import PointerActorCritic

        student = PointerActorCritic(sindex, NUM_ACTIONS)
    else:
        student = ActorCritic(sindex, NUM_ACTIONS, encoder=args.student_encoder)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    n = len(acts)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot, correct, losses = 0, 0, []
        for s in range(0, n, args.batch):
            mb = perm[s:s + args.batch]
            dist, _ = student.dist_value(obs_n[mb], masks[mb])
            loss = F.nll_loss(torch.log(dist.probs.clamp(min=1e-9)), acts[mb])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            correct += (dist.probs.argmax(-1) == acts[mb]).sum().item()
            tot += len(mb)
        print(f"epoch {ep+1}: CE {np.mean(losses):.3f} | teacher-match {correct/tot:.1%}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ckpt_00000.pt")
    torch.save(dict(model=clean_state_dict(student), norm=norm.state_dict(),
                    iter=0, encoder=args.student_encoder, derived=derived), path)
    print(f"saved warm-start {path} -> fine-tune with: "
          f"train --encoder {args.student_encoder}"
          f"{' --derived' if derived else ''} --resume --out {args.out}")


if __name__ == "__main__":
    main()
