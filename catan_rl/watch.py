"""Visualize games as a self-contained HTML replay.

Plays one full game with a trained checkpoint and records every action by
every player (not just the agent's decisions) through catanatron's
GameAccumulator hooks. Output is a single HTML file with an SVG hex board,
a turn slider, and per-player stats. No dependencies; scp it down or open
it through VS Code Remote / `python -m http.server`.

Usage:
  python -m catan_rl.watch --ckpt checkpoints/seed0/ckpt_00100.pt \
      --opponents self --out replay.html
  --opponents self      all four seats play the same checkpoint (true self-play)
  --opponents weighted  BLUE = checkpoint, others WeightedRandomPlayer
  --opponents random    BLUE = checkpoint, others RandomPlayer
"""

import argparse
import json
import math
import os

import torch
from catanatron.game import Game, GameAccumulator
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import (
    get_actual_victory_points,
    get_largest_army,
    get_longest_road_color,
    player_num_resource_cards,
)

from .env import DERIVED_SIZE, CatanEnv, RunningNorm
from .model import ActorCritic
from .opponents import NNPlayer
from .streams import build_stream_index

NUM_ACTIONS = 290
SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]
SQRT3 = math.sqrt(3)

# NodeRef corner offsets for a pointy-top hex, SVG coords (y down), unit size
CORNERS = {
    "NORTH": (0.0, -1.0),
    "NORTHEAST": (SQRT3 / 2, -0.5),
    "SOUTHEAST": (SQRT3 / 2, 0.5),
    "SOUTH": (0.0, 1.0),
    "SOUTHWEST": (-SQRT3 / 2, 0.5),
    "NORTHWEST": (-SQRT3 / 2, -0.5),
}

RESOURCE_FILL = {
    "WOOD": "#2f7d32",
    "BRICK": "#b5502a",
    "SHEEP": "#9ccc65",
    "WHEAT": "#e2b93b",
    "ORE": "#8d99a6",
    None: "#d8c7a1",  # desert
}


def hex_center(coord, size):
    q, _, r = coord  # cube -> axial (q = x, r = z)
    return (size * SQRT3 * (q + r / 2.0), size * 1.5 * r)


def board_geometry(game, size=52):
    tiles, node_pos = [], {}
    coord_index = {}
    for i, (coord, tile) in enumerate(game.state.board.map.land_tiles.items()):
        cx, cy = hex_center(coord, size)
        coord_index[coord] = i
        tiles.append(
            dict(
                cx=cx,
                cy=cy,
                fill=RESOURCE_FILL[tile.resource],
                resource=tile.resource or "DESERT",
                number=tile.number,
            )
        )
        for ref, node_id in tile.nodes.items():
            dx, dy = CORNERS[ref.value]
            node_pos.setdefault(node_id, (cx + dx * size, cy + dy * size))
    return tiles, node_pos, coord_index, size


def snapshot(game, coord_index):
    b = game.state.board
    buildings = [
        [int(n), c.value, t] for n, (c, t) in sorted(b.buildings.items())
    ]
    roads = sorted(
        {(tuple(sorted(e)), c.value) for e, c in b.roads.items()}
    )
    la_color, _ = get_largest_army(game.state)
    players = []
    for c in SEAT_COLORS:
        players.append(
            dict(
                color=c.value,
                vp=get_actual_victory_points(game.state, c),
                cards=player_num_resource_cards(game.state, c),
                lr=get_longest_road_color(game.state) == c,
                la=la_color == c,
            )
        )
    return dict(
        buildings=buildings,
        roads=[[list(e), c] for e, c in roads],
        robber=coord_index[b.robber_coordinate],
        players=players,
        turn=game.state.num_turns,
    )


class ReplayAccumulator(GameAccumulator):
    def __init__(self, coord_index):
        self.coord_index = coord_index
        self.frames = []
        self._pending = None

    def _describe(self, action):
        v = action.value
        return f"{action.color.value} {action.action_type.value}" + (
            f" {v}" if v is not None else ""
        )

    def step(self, game_before_action, action):
        if self._pending is not None:
            self.frames.append(
                dict(action=self._pending, state=snapshot(game_before_action, self.coord_index))
            )
        self._pending = self._describe(action)

    def after(self, game):
        if self._pending is not None:
            self.frames.append(
                dict(action=self._pending, state=snapshot(game, self.coord_index))
            )
        winner = game.winning_color()
        self.winner = winner.value if winner else None


def load_agent(ckpt_path):
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    derived = data.get("derived", False)
    sindex = build_stream_index(4, num_derived=DERIVED_SIZE if derived else 0)
    enc = data.get("encoder", "flat")
    if enc == "pointer":
        from .pointer_model import PointerActorCritic

        model = PointerActorCritic(sindex, NUM_ACTIONS)
    else:
        model = ActorCritic(sindex, NUM_ACTIONS, encoder=enc)
    model.load_state_dict(data["model"])
    model.eval()
    probe = CatanEnv(derived=derived)
    norm = RunningNorm(probe.obs_size)
    norm.load_state_dict(data["norm"])
    norm.frozen = True
    del probe
    model.uses_derived = derived
    calib_path = ckpt_path + ".calib.json"
    if os.path.exists(calib_path):
        import json

        with open(calib_path) as f:
            model.calibration = json.load(f)
    else:
        model.calibration = None
    return model, norm


def build_players(model, norm, opponents, deterministic, trade=False,
                  trade_log=None):
    def nn(color):
        return NNPlayer(color, model, norm, deterministic=deterministic,
                        derived=getattr(model, "uses_derived", False),
                        trade=trade, trade_log=trade_log)

    if opponents == "self":
        return [nn(c) for c in SEAT_COLORS]
    enemy = WeightedRandomPlayer if opponents == "weighted" else RandomPlayer
    return [nn(SEAT_COLORS[0])] + [enemy(c) for c in SEAT_COLORS[1:]]


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Catan RL Replay</title>
<style>
 body{font-family:system-ui,sans-serif;background:#1c1f26;color:#e8e8e8;margin:0;display:flex;flex-direction:column;align-items:center}
 h2{margin:14px 0 4px} .sub{color:#9aa4b2;margin-bottom:8px;font-size:14px}
 #wrap{display:flex;gap:24px;align-items:flex-start;padding:12px}
 svg{background:#26505f;border-radius:12px}
 #panel{width:300px}
 .p{border-radius:8px;padding:8px 10px;margin-bottom:8px;background:#2a2e37;border-left:6px solid}
 .p b{font-size:15px} .stat{font-size:13px;color:#c8d0da}
 #action{min-height:40px;font-size:14px;background:#2a2e37;border-radius:8px;padding:8px 10px;margin-bottom:10px}
 #controls{display:flex;gap:8px;align-items:center;margin:10px 0}
 button{background:#3b4252;color:#e8e8e8;border:0;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:14px}
 button:hover{background:#4c566a}
 input[type=range]{width:420px}
 .badge{font-size:11px;background:#e2b93b;color:#222;border-radius:4px;padding:1px 5px;margin-left:6px}
</style></head><body>
<h2>Catan RL Replay</h2>
<div class="sub">__SUBTITLE__</div>
<div id="controls">
 <button onclick="jump(0)">&#171;</button>
 <button onclick="stepBy(-1)">&#8249;</button>
 <button id="playbtn" onclick="togglePlay()">Play</button>
 <button onclick="stepBy(1)">&#8250;</button>
 <button onclick="jump(F.length-1)">&#187;</button>
 <input type="range" id="slider" min="0" max="0" value="0" oninput="jump(+this.value)">
 <span id="counter"></span>
</div>
<div id="wrap">
 <svg id="board" width="640" height="600" viewBox="__VIEWBOX__"></svg>
 <div id="panel"><div id="action"></div><div id="players"></div></div>
</div>
<script>
const GEO = __GEO__;
const SHOWIDS = __SHOWIDS__;
const F = __FRAMES__;
const WINNER = __WINNER__;
const PCOLOR = {BLUE:"#3b7dd8",RED:"#d64545",ORANGE:"#e08a2e",WHITE:"#e8e8e8"};
let i = 0, timer = null;
const svg = document.getElementById("board");
function el(t,a){const e=document.createElementNS("http://www.w3.org/2000/svg",t);for(const k in a)e.setAttribute(k,a[k]);return e;}
function hexPts(cx,cy,s){const p=[];for(let k=0;k<6;k++){const a=Math.PI/180*(60*k-90);p.push((cx+s*Math.cos(a))+","+(cy+s*Math.sin(a)));}return p.join(" ");}
function draw(){
 svg.innerHTML="";
 const st=F[i].state, s=GEO.size;
 GEO.tiles.forEach((t,ti)=>{
  svg.appendChild(el("polygon",{points:hexPts(t.cx,t.cy,s-1),fill:t.fill,stroke:"#1c1f26","stroke-width":2}));
  if(t.number){svg.appendChild(el("circle",{cx:t.cx,cy:t.cy,r:13,fill:"#f4ecd8"}));
   const tx=el("text",{x:t.cx,y:t.cy+5,"text-anchor":"middle","font-size":14,"font-weight":700,
    fill:(t.number==6||t.number==8)?"#c0392b":"#333"});tx.textContent=t.number;svg.appendChild(tx);}
  if(st.robber===ti){svg.appendChild(el("circle",{cx:t.cx+16,cy:t.cy-16,r:9,fill:"#111",stroke:"#e8e8e8","stroke-width":1.5}));}
 });
 st.roads.forEach(([e,c])=>{const a=GEO.nodes[e[0]],b=GEO.nodes[e[1]];
  svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:PCOLOR[c],"stroke-width":7,"stroke-linecap":"round"}));});
 if(SHOWIDS){for(const [nid,p] of Object.entries(GEO.nodes)){
   const t=el("text",{x:p[0]+11,y:p[1]-8,"font-size":10,fill:"#cfd8e3","text-anchor":"middle"});t.textContent=nid;svg.appendChild(t);}}
 st.buildings.forEach(([n,c,t])=>{const p=GEO.nodes[n];
  if(t==="CITY"){svg.appendChild(el("rect",{x:p[0]-9,y:p[1]-9,width:18,height:18,fill:PCOLOR[c],stroke:"#111","stroke-width":1.5}));}
  else{svg.appendChild(el("circle",{cx:p[0],cy:p[1],r:8,fill:PCOLOR[c],stroke:"#111","stroke-width":1.5}));}});
 document.getElementById("action").innerHTML =
   "<b>Turn "+st.turn+"</b> &mdash; "+F[i].action+(i===F.length-1&&WINNER?("<br><b style='color:"+PCOLOR[WINNER]+"'>&#127942; "+WINNER+" wins</b>"):"");
 document.getElementById("players").innerHTML = st.players.map(p=>
   "<div class='p' style='border-color:"+PCOLOR[p.color]+"'><b style='color:"+PCOLOR[p.color]+"'>"+p.color+"</b>"+
   (p.lr?"<span class='badge'>Longest Road</span>":"")+(p.la?"<span class='badge'>Largest Army</span>":"")+
   "<div class='stat'>VP: "+p.vp+" &nbsp; cards: "+p.cards+"</div></div>").join("");
 document.getElementById("slider").value=i;
 document.getElementById("counter").textContent=(i+1)+" / "+F.length;
}
function jump(k){i=Math.max(0,Math.min(F.length-1,k));draw();}
function stepBy(d){jump(i+d);}
function togglePlay(){
 const b=document.getElementById("playbtn");
 if(timer){clearInterval(timer);timer=null;b.textContent="Play";return;}
 b.textContent="Pause";
 timer=setInterval(()=>{if(i>=F.length-1){togglePlay();}else{stepBy(1);}},120);
}
document.getElementById("slider").max=F.length-1;
document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")stepBy(1);if(e.key==="ArrowLeft")stepBy(-1);if(e.key===" "){e.preventDefault();togglePlay();}});
jump(0);
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--opponents", choices=["self", "weighted", "random"], default="self")
    p.add_argument("--out", default="replay.html")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    model, norm = load_agent(args.ckpt)
    players = build_players(model, norm, args.opponents, deterministic=not args.stochastic)

    game = Game(players, seed=args.seed)
    tiles, node_pos, coord_index, size = board_geometry(game)
    acc = ReplayAccumulator(coord_index)
    game.play(accumulators=[acc])

    xs = [t["cx"] for t in tiles]
    ys = [t["cy"] for t in tiles]
    pad = size * 1.4
    viewbox = (
        f"{min(xs) - pad:.0f} {min(ys) - pad:.0f} "
        f"{max(xs) - min(xs) + 2 * pad:.0f} {max(ys) - min(ys) + 2 * pad:.0f}"
    )

    geo = dict(
        tiles=tiles,
        nodes={str(k): [round(v[0], 1), round(v[1], 1)] for k, v in node_pos.items()},
        size=size,
    )
    subtitle = (
        f"checkpoint: {os.path.basename(args.ckpt)} | opponents: {args.opponents} | "
        f"{len(acc.frames)} actions | winner: {acc.winner or 'none (turn limit)'}"
    )
    html = (
        HTML_TEMPLATE.replace("__GEO__", json.dumps(geo))
        .replace("__FRAMES__", json.dumps(acc.frames))
        .replace("__WINNER__", json.dumps(acc.winner))
        .replace("__VIEWBOX__", viewbox)
        .replace("__SUBTITLE__", subtitle)
        .replace("__SHOWIDS__", "false")
    )
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}: {subtitle}")
    if tlog is not None:
        print("trading:", tlog.summary())


if __name__ == "__main__":
    main()
