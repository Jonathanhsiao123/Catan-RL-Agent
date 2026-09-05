"""Play against the trained agent on an interactive board in your browser.

You are BLUE; the checkpoint plays the other three seats.

Board moves are clickable highlights; everything else is a button. With
--trade you are a full participant in the trading system: bots offer you
trades (you accept or reject), and you can build and send your own offers,
which the bots price with their critic.

Usage:
  python -m catan_rl.play --ckpt checkpoints/gen3/ckpt_08000.pt --port 8377 --trade

Then from your laptop:  ssh -L 8377:localhost:8377 <user>@<server>
and open http://localhost:8377
"""

import argparse
import json
import os
import queue
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from catanatron.game import Game
from catanatron.models.player import Color, Player

from .trading import (
    RESOURCES,
    TradeConfig,
    TradeLog,
    human_propose,
    is_trade_window,
)
from .watch import ReplayAccumulator, board_geometry, build_players, load_agent, snapshot

SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]


# ---------------------------------------------------------------- shared state
class Bridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.choice = queue.Queue()        # human action / propose requests
        self.trade_reply = queue.Queue()   # human accept/reject of a bot offer
        self.snapshot = None
        self.log = []
        self.pending = None                # {turn_id, actions}
        self.pending_trade = None          # {id, proposer, gives, wants}
        self.turn_id = 0
        self.trade_id = 0
        self.hand = {}
        self.trades = []
        self.trade_result = None
        self.can_propose = False
        self.notice = None
        self.winner = None
        self.done = False

    def view(self):
        with self.lock:
            return dict(
                snapshot=self.snapshot, log=self.log[-14:],
                pending=self.pending, hand=self.hand,
                pending_trade=self.pending_trade,
                can_propose=self.can_propose,
                trade_result=self.trade_result,
                trades=self.trades[-6:],
                notice=self.notice,
                winner=self.winner, done=self.done,
            )


# ------------------------------------------------------------- action labeling
def group_combo_actions(actions):
    """Collapse many near-identical PLAY_YEAR_OF_PLENTY/PLAY_MONOPOLY entries
    (one per resource combination, per catanatron's action enumeration) into a
    single card per type, carrying every legal option for a picker to choose
    from. Without this a hand with an unrestricted bank produces 15+ nearly
    identical "Year of Plenty" entries."""
    grouped, rest = {}, []
    for a in actions:
        if a["kind"] != "combo":
            rest.append(a)
            continue
        g = grouped.setdefault(a["card"], dict(
            kind="card", card=a["card"], label=a["card"], options=[]))
        g["options"].append(dict(i=a["i"], combo=a["combo"], label=a["label"]))
    return rest + list(grouped.values())


def describe_actions(actions, node_prod, coord_index):
    """Turn engine actions into clickable/button payloads for the browser."""
    out = []
    for i, a in enumerate(actions):
        t = a.action_type.value
        v = a.value
        if t in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            prod = node_prod.get(v, {})
            pips = ", ".join(
                f"{r.lower()}\u2022{round(p * 36)}" for r, p in
                sorted(prod.items(), key=lambda kv: -kv[1]))
            word = "Settlement" if t == "BUILD_SETTLEMENT" else "City"
            out.append(dict(i=i, kind="node", node=v,
                            city=(t == "BUILD_CITY"),
                            label=f"{word} ({pips})" if pips else word))
        elif t == "BUILD_ROAD":
            out.append(dict(i=i, kind="edge", edge=list(v), label="Road"))
        elif t == "MOVE_ROBBER":
            coord, victim = v[0], v[1]
            out.append(dict(i=i, kind="hex", tile=coord_index[coord],
                            victim=victim.value if victim else None,
                            label="Move robber"))
        elif t == "MARITIME_TRADE":
            give = [r for r in v[:4] if r is not None]
            out.append(dict(i=i, kind="button",
                            label=f"Bank: {len(give)} {give[0].lower()} \u2192 1 {v[4].lower()}"))
        elif t == "PLAY_YEAR_OF_PLENTY":
            take = sorted(r for r in v if r)
            out.append(dict(i=i, kind="combo", card="YEAR_OF_PLENTY",
                            combo=take, label="take " + " + ".join(
                                r.lower() for r in take)))
        elif t == "PLAY_MONOPOLY":
            out.append(dict(i=i, kind="combo", card="MONOPOLY",
                            combo=[v], label=f"claim all {v.lower()}"))
        elif t == "PLAY_KNIGHT_CARD":
            out.append(dict(i=i, kind="card", card="KNIGHT", label="move the robber"))
        elif t == "PLAY_ROAD_BUILDING":
            out.append(dict(i=i, kind="card", card="ROAD_BUILDING",
                            label="2 free roads"))
        elif t == "BUY_DEVELOPMENT_CARD":
            out.append(dict(i=i, kind="button", label="Buy dev card"))
        else:
            label = {"ROLL": "Roll dice", "END_TURN": "End turn",
                     "DISCARD": "Discard (engine picks)"}.get(t, t)
            out.append(dict(i=i, kind="button", label=label))
    return out


# ------------------------------------------------------------------- the human
class WebHumanPlayer(Player):
    def __init__(self, color, bridge, node_prod, coord_index,
                 trade_config=None, trade_log=None):
        super().__init__(color)
        self.bridge = bridge
        self.node_prod = node_prod
        self.coord_index = coord_index
        self.trade_config = trade_config or TradeConfig()
        self.trade_log = trade_log
        self._prev_dev_counts = None
        self._bought_this_turn = {}
        self._last_turn_seen = -1

    def _hand(self, game):
        idx = game.state.color_to_index[self.color]
        ps = game.state.player_state
        return {r: ps.get(f"P{idx}_{r}_IN_HAND", 0) for r in RESOURCES}

    def _dev_counts(self, game):
        idx = game.state.color_to_index[self.color]
        ps = game.state.player_state
        cards = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING",
                 "VICTORY_POINT")
        return {c: ps.get(f"P{idx}_{c}_IN_HAND", 0) for c in cards}

    def _playable_now(self, card_type):
        """How many copies of card_type were NOT bought this turn (house rule:
        the engine allows playing a card the turn it's bought; we hold the
        human to the stricter, physical-game standard)."""
        total = self._prev_dev_counts.get(card_type, 0) if self._prev_dev_counts else 0
        bought = self._bought_this_turn.get(card_type, 0)
        return max(0, total - bought)

    # ---- called by trading.try_trade from the proposing bot's thread ----
    def ask_trade(self, game, proposer, offer):
        b = self.bridge
        with b.lock:
            b.trade_id += 1
            b.hand = self._hand(game)
            b.pending_trade = dict(
                id=b.trade_id, proposer=proposer.value,
                gives={k: v for k, v in offer.give.items() if v},
                wants={k: v for k, v in offer.ask.items() if v},
                text=str(offer))
        while True:
            tid, accept = b.trade_reply.get()
            if tid == b.trade_id:
                break
        with b.lock:
            b.pending_trade = None
            b.hand = self._hand(game)
        return bool(accept)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        b = self.bridge

        # ---- 2/5. detect what changed since our last decision: a dev card
        # bought (and which type), so we can (a) notify even for silent
        # Victory Point cards, and (b) enforce "can't play a card the same
        # turn you bought it" for the human, since the engine itself does
        # not (player_can_play_dev only checks total count, not vintage).
        cur_dev = self._dev_counts(game)
        turn = game.state.num_turns
        if turn != self._last_turn_seen:
            self._last_turn_seen = turn
            self._bought_this_turn = {}
        if self._prev_dev_counts is not None:
            for card, n in cur_dev.items():
                gained = n - self._prev_dev_counts.get(card, 0)
                if gained > 0:
                    self._bought_this_turn[card] = (
                        self._bought_this_turn.get(card, 0) + gained)
                    with b.lock:
                        b.notice = ("Drew a hidden Victory Point card (+1 VP, "
                                   "kept secret from opponents)"
                                   if card == "VICTORY_POINT"
                                   else f"Drew a {card.replace('_',' ').title()} card")
        self._prev_dev_counts = cur_dev

        with b.lock:
            b.turn_id += 1
            b.hand = self._hand(game)
            b.can_propose = is_trade_window(game, self.color)
            # 1. board was one action stale; this is the TRUE current state
            b.snapshot = snapshot(game, self.coord_index)
            actions = group_combo_actions(describe_actions(
                playable_actions, self.node_prod, self.coord_index))
            for a in actions:
                if a["kind"] == "card":
                    a["locked"] = self._playable_now(a["card"]) <= 0
            b.pending = dict(turn_id=b.turn_id, actions=actions)
        while True:
            msg = b.trade_reply if False else b.choice.get()
            kind = msg[0]
            if kind == "propose":
                # runs on the game thread, so mutating state here is safe
                _, give, ask = msg
                res = human_propose(game, self.color, give, ask,
                                    self.trade_config, self.trade_log)
                with b.lock:
                    b.trade_result = (
                        f"{res['accepter']} accepted: {res['offer']}"
                        if res.get("ok") else f"No deal \u2014 {res['reason']}")
                    b.hand = self._hand(game)
                    b.can_propose = is_trade_window(game, self.color)
                    # hand changed: refresh the legal set and the UI payload
                    playable_actions = game.state.playable_actions
                    b.turn_id += 1
                    actions = group_combo_actions(describe_actions(
                        playable_actions, self.node_prod, self.coord_index))
                    for a in actions:
                        if a["kind"] == "card":
                            a["locked"] = self._playable_now(a["card"]) <= 0
                    b.pending = dict(turn_id=b.turn_id, actions=actions)
                continue
            _, turn_id, idx = msg
            if turn_id == b.turn_id and 0 <= idx < len(playable_actions):
                break
        with b.lock:
            b.pending = None
            b.can_propose = False
        return playable_actions[idx]


class LiveAccumulator(ReplayAccumulator):
    def __init__(self, coord_index, bridge):
        super().__init__(coord_index)
        self.bridge = bridge

    def step(self, game_before_action, action):
        super().step(game_before_action, action)
        if self.frames:
            with self.bridge.lock:
                self.bridge.snapshot = self.frames[-1]["state"]
                self.bridge.log = [f["action"] for f in self.frames]

    def after(self, game):
        super().after(game)
        with self.bridge.lock:
            if self.frames:
                self.bridge.snapshot = self.frames[-1]["state"]
                self.bridge.log = [f["action"] for f in self.frames]
            self.bridge.winner = self.winner
            self.bridge.done = True


# ---------------------------------------------------------------------- server
def make_handler(bridge, page, token=""):
    prefix = f"/{token}" if token else ""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _route(self):
            if token:
                if not self.path.startswith(prefix):
                    return None
                return self.path[len(prefix):] or "/"
            return self.path

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self._route()
            if route is None:
                return self._json({"err": "not found"}, 404)
            if route == "/state":
                return self._json(bridge.view())
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            route = self._route()
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            if route == "/act":
                bridge.choice.put(("action", int(data.get("turn_id", -1)),
                                   int(data.get("i", -1))))
                return self._json({"ok": True})
            if route == "/trade":
                bridge.trade_reply.put((int(data.get("id", -1)),
                                        bool(data.get("accept"))))
                return self._json({"ok": True})
            if route == "/propose":
                give = {r: int(data.get("give", {}).get(r, 0)) for r in RESOURCES}
                ask = {r: int(data.get("ask", {}).get(r, 0)) for r in RESOURCES}
                bridge.choice.put(("propose", give, ask))
                return self._json({"ok": True})
            return self._json({"err": "bad path"}, 404)

    return H


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Catan vs RL Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap" rel="stylesheet">
<style>
 :root{--ink:#0c1a20;--panel:#1c2429;--panel2:#232c32;--line:rgba(255,255,255,.08);
  --text:#eae7e0;--dim:#95a2ab;--go:#a3e635;--warn:#e8b04b}
 body{font-family:system-ui,sans-serif;background:var(--ink);color:var(--text);margin:0}
 #head{display:flex;justify-content:center;gap:10px;align-items:baseline;padding:14px 0 2px}
 #head b{font-family:"Fraunces",Georgia,serif;font-size:22px}
 #head span{color:var(--dim);font-size:13px}
 #wrap{display:flex;justify-content:center;gap:20px;padding:12px 16px;align-items:flex-start;flex-wrap:wrap}
 svg{border-radius:18px;box-shadow:0 10px 34px rgba(0,0,0,.5)}
 #panel{width:336px} #ref{width:300px}
 h3{margin:14px 0 6px;font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.09em}
 #status{font-family:"Fraunces",Georgia,serif;font-size:18px;background:var(--panel);
  border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px}
 #status.mine{background:#2a4527;border-color:#3f6b39}
 .p{border-radius:8px;padding:6px 10px;margin-bottom:6px;background:var(--panel);
  border:1px solid var(--line);border-left:6px solid;display:flex;justify-content:space-between;font-size:14px}
 .chip{display:inline-block;border-radius:6px;padding:3px 8px;margin:2px;font-size:13px;font-weight:600}
 .r-WOOD{background:#2f7d32;color:#fff}.r-BRICK{background:#b5502a;color:#fff}
 .r-SHEEP{background:#9ccc65;color:#1c2b12}.r-WHEAT{background:#e2b93b;color:#3a2d08}
 .r-ORE{background:#8d99a6;color:#1a2027}
 .mini{padding:1px 6px;font-size:11px;margin:1px}
 button{font-family:inherit}
 #buttons button{display:block;width:100%;text-align:left;background:var(--panel2);color:var(--text);
  border:1px solid var(--line);border-radius:8px;padding:9px 11px;margin:5px 0;cursor:pointer;font-size:14px}
 #buttons button:hover{background:#2e3a42;border-color:#48565f}
 #buttons .none{color:var(--dim);font-size:13px;padding:6px 2px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px}
 .cost{display:flex;justify-content:space-between;align-items:center;margin:6px 0;font-size:14px}
 .cost .vp{color:var(--dim);font-size:12px;margin-left:6px}
 #log{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--dim);
  max-height:230px;overflow-y:auto;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:10px;line-height:1.55}
 #offer{background:#3a2f1c;border:1px solid #6b5525;border-radius:12px;padding:12px 14px;
  margin-bottom:12px;display:none}
 #offer .who{font-family:"Fraunces",Georgia,serif;font-size:16px;margin-bottom:8px}
 #offer .row{margin:5px 0;font-size:13px}
 #offer .btns{display:flex;gap:8px;margin-top:10px}
 #offer button{flex:1;border:0;border-radius:8px;padding:9px;cursor:pointer;font-size:14px;font-weight:600}
 #yes{background:var(--go);color:#12240b}#no{background:#4a3a3a;color:#f0e6e6}
 #builder table{width:100%;font-size:13px;border-collapse:collapse}
 #builder td{padding:3px 2px}
 #builder .stepper{display:flex;gap:4px;align-items:center;justify-content:flex-end}
 #builder .stepper button{width:22px;height:22px;border-radius:5px;border:1px solid var(--line);
  background:var(--panel2);color:var(--text);cursor:pointer;line-height:1;padding:0}
 #builder .n{min-width:14px;text-align:center;font-variant-numeric:tabular-nums}
 #send{width:100%;margin-top:10px;background:var(--go);color:#12240b;border:0;border-radius:8px;
  padding:10px;font-weight:700;cursor:pointer;font-size:14px}
 #send:disabled{background:#3a4048;color:#7b858d;cursor:not-allowed}
 #tresult{font-size:12px;color:var(--warn);margin-top:8px;min-height:16px}
 #devcards{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 4px}
 .devcard{width:92px;height:128px;border-radius:10px;position:relative;cursor:pointer;
  color:#2a1f10;font-size:11px;font-weight:700;text-align:center;
  box-shadow:0 4px 10px rgba(0,0,0,.5);border:2px solid rgba(0,0,0,.35);
  display:flex;flex-direction:column;justify-content:space-between;padding:7px 5px;
  transition:transform .12s ease;overflow:hidden}
 .devcard:hover{transform:translateY(-5px) rotate(-1deg)}
 .devcard .icon{font-size:30px;margin-top:6px}
 .devcard .name{font-family:"Fraunces",Georgia,serif;font-size:12.5px;line-height:1.15}
 .devcard .sub{font-size:9.5px;font-weight:500;opacity:.85}
 .devcard::before{content:"";position:absolute;inset:5px;border:1px solid rgba(0,0,0,.25);
  border-radius:6px;pointer-events:none}
 .devcard.locked{cursor:default;filter:grayscale(75%) brightness(.62);opacity:.85}
 .devcard.locked:hover{transform:none}
 .devcard.locked .sub{font-style:italic}
 .dc-KNIGHT{background:linear-gradient(160deg,#c8752f,#a85420)}
 .dc-YEAR_OF_PLENTY{background:linear-gradient(160deg,#e0c355,#b89424)}
 .dc-MONOPOLY{background:linear-gradient(160deg,#c85a5a,#963636)}
 .dc-ROAD_BUILDING{background:linear-gradient(160deg,#7fae63,#4d7c3a)}
 #picker{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;
  align-items:center;justify-content:center;z-index:50}
 #picker .box{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:18px 20px;width:300px}
 #picker h4{font-family:"Fraunces",Georgia,serif;margin:0 0 12px;font-size:17px}
 #picker .opt{display:block;width:100%;text-align:left;background:var(--panel2);
  color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 12px;
  margin:5px 0;cursor:pointer;font-size:14px}
 #picker .opt:hover{background:#2e3a42;border-color:#48565f}
 #picker .cancel{margin-top:8px;background:none;border:0;color:var(--dim);
  cursor:pointer;font-size:12px}
 #notice{background:#2a3d4d;border:1px solid #3d6a86;border-radius:10px;
  padding:9px 12px;margin-bottom:10px;font-size:13px;display:none}
 .clicky{cursor:pointer;animation:soft 2.2s ease-in-out infinite}
 @keyframes soft{0%,100%{opacity:.9}50%{opacity:.55}}
 @media (prefers-reduced-motion: reduce){.clicky{animation:none;opacity:.85}}
 .tile:hover{filter:brightness(1.09)}
</style></head><body>
<div id="picker"><div class="box"><h4 id="pk-title"></h4><div id="pk-opts"></div><button class="cancel" onclick="closePicker()">cancel</button></div></div>
<div id="head"><b>Catan</b><span>you (blue) vs three copies of __CKPT__</span></div>
<div id="wrap">
 <svg id="board" width="720" height="690" viewBox="__VIEWBOX__"></svg>
 <div id="panel">
  <div id="notice"></div>
  <div id="status">Connecting to the game&hellip;</div>
  <div id="offer">
   <div class="who"></div>
   <div class="row give"></div>
   <div class="row want"></div>
   <div class="btns"><button id="yes">Accept</button><button id="no">Reject</button></div>
  </div>
  <div id="players"></div>
  <h3>Your hand</h3><div id="hand"></div>
  <div id="victim"></div>
  <h3>Development cards</h3><div id="devcards"></div>
  <h3>Actions</h3><div id="buttons"></div>
 </div>
 <div id="ref">
  <h3 style="margin-top:0">Propose a trade</h3>
  <div class="card" id="builder">
   <table id="btbl"></table>
   <button id="send" onclick="sendOffer()">Send offer to the table</button>
   <div id="tresult"></div>
  </div>
  <div class="card">
   <h3 style="margin-top:0">Building costs</h3>
   <div class="cost"><span>Road</span><span><span class="chip mini r-WOOD">wood</span><span class="chip mini r-BRICK">brick</span></span></div>
   <div class="cost"><span>Settlement <span class="vp">1 VP</span></span><span><span class="chip mini r-WOOD">wood</span><span class="chip mini r-BRICK">brick</span><span class="chip mini r-SHEEP">sheep</span><span class="chip mini r-WHEAT">wheat</span></span></div>
   <div class="cost"><span>City <span class="vp">2 VP</span></span><span><span class="chip mini r-WHEAT">wheat</span><span class="chip mini r-WHEAT">wheat</span><span class="chip mini r-ORE">ore</span><span class="chip mini r-ORE">ore</span><span class="chip mini r-ORE">ore</span></span></div>
   <div class="cost"><span>Dev card</span><span><span class="chip mini r-SHEEP">sheep</span><span class="chip mini r-WHEAT">wheat</span><span class="chip mini r-ORE">ore</span></span></div>
   <div class="cost" style="color:var(--dim);font-size:12px"><span>Longest Road / Largest Army</span><span>2 VP each &middot; first to 10 wins</span></div>
  </div>
  <h3>Trades</h3><div class="card" id="trades" style="font-size:12px;color:#c8d0da">none yet</div>
  <h3>Game log</h3><div id="log"><span style="color:var(--dim)">Nothing yet.</span></div>
 </div>
</div>
<script>
const BASE = "__BASE__";
const GEO = __GEO__;
const RES = ["WOOD","BRICK","SHEEP","WHEAT","ORE"];
const PCOLOR = {BLUE:"#4a8fe7",RED:"#e05252",ORANGE:"#e8952e",WHITE:"#f0ece4"};
const DEVCARD = {
 KNIGHT:{icon:"\u2694\ufe0f",name:"Knight"},
 YEAR_OF_PLENTY:{icon:"\ud83c\udf3e",name:"Year of\nPlenty"},
 MONOPOLY:{icon:"\ud83d\udc51",name:"Monopoly"},
 ROAD_BUILDING:{icon:"\ud83d\udee3\ufe0f",name:"Road\nBuilding"},
};
const GRAD = {WOOD:["#41a049","#1f5c26"],BRICK:["#d3703f","#8c3d1e"],
 SHEEP:["#bce285","#78a94a"],WHEAT:["#f2cf58","#c39a26"],
 ORE:["#a8b4c0","#66717d"],DESERT:["#e8d9b0","#c0a97e"]};
const svg = document.getElementById("board");
const CTR = (()=>{let x=0,y=0;GEO.tiles.forEach(t=>{x+=t.cx;y+=t.cy});
 return [x/GEO.tiles.length,y/GEO.tiles.length];})();
let lastPaint="", offerShown=0;
const give={}, ask={};
RES.forEach(r=>{give[r]=0;ask[r]=0;});

function el(t,a,title){const e=document.createElementNS("http://www.w3.org/2000/svg",t);
 for(const k in a)e.setAttribute(k,a[k]);
 if(title){const ti=document.createElementNS("http://www.w3.org/2000/svg","title");
  ti.textContent=title;e.appendChild(ti);}
 return e;}
function hexPts(cx,cy,s){const p=[];for(let k=0;k<6;k++){const a=Math.PI/180*(60*k-90);
 p.push((cx+s*Math.cos(a))+","+(cy+s*Math.sin(a)));}return p.join(" ");}
function post(path,body){return fetch(BASE+path,{method:"POST",
 headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});}
function act(turn_id,i){post("/act",{turn_id:turn_id,i:i});
 document.getElementById("victim").style.display="none";}

function defs(){
 const d=el("defs",{});
 for(const r in GRAD){
  const g=el("radialGradient",{id:"g"+r,cx:"38%",cy:"32%",r:"82%"});
  g.appendChild(el("stop",{offset:"0%","stop-color":GRAD[r][0]}));
  g.appendChild(el("stop",{offset:"100%","stop-color":GRAD[r][1]}));
  d.appendChild(g);
 }
 const wp=el("pattern",{id:"waves",width:26,height:14,patternUnits:"userSpaceOnUse"});
 wp.appendChild(el("path",{d:"M0 7 Q6.5 2 13 7 T26 7",stroke:"rgba(255,255,255,.07)",
  "stroke-width":1.4,fill:"none"}));
 wp.appendChild(el("path",{d:"M0 11 Q6.5 6 13 11 T26 11",stroke:"rgba(255,255,255,.045)",
  "stroke-width":1.2,fill:"none"}));
 d.appendChild(wp);
 // subtle per-resource texture so tiles read as terrain, not flat polygons
 const tex={WOOD:"M0 6 L6 0 M-2 2 L2 -2",BRICK:"M0 0 H8 M0 5 H8 M4 0 V5",
  SHEEP:"M2 2 m-1 0 a1 1 0 1 0 2 0 a1 1 0 1 0 -2 0",WHEAT:"M1 6 V1 M4 6 V2",
  ORE:"M1 5 L3 1 L5 5 Z",DESERT:"M0 4 H6"};
 for(const r in tex){
  const p=el("pattern",{id:"t"+r,width:8,height:8,patternUnits:"userSpaceOnUse",
   patternTransform:"rotate(18)"});
  p.appendChild(el("path",{d:tex[r],stroke:"rgba(255,255,255,.16)","stroke-width":1,fill:"none"}));
  d.appendChild(p);
 }
 const f=el("filter",{id:"drop",x:"-40%",y:"-40%",width:"180%",height:"180%"});
 f.appendChild(el("feDropShadow",{dx:0,dy:2,stdDeviation:2,"flood-opacity":.55}));
 d.appendChild(f);
 const w=el("radialGradient",{id:"water",cx:"50%",cy:"42%",r:"78%"});
 w.appendChild(el("stop",{offset:"0%","stop-color":"#2f6a80"}));
 w.appendChild(el("stop",{offset:"70%","stop-color":"#1d4655"}));
 w.appendChild(el("stop",{offset:"100%","stop-color":"#122c37"}));
 d.appendChild(w);
 return d;}

function defsWrap(node){const d=el("defs",{});d.appendChild(node);return d;}
function housePath(x,y,s){ // settlement: little house
 return `M${x-s} ${y+s} L${x-s} ${y-s*0.2} L${x} ${y-s*1.1} L${x+s} ${y-s*0.2} L${x+s} ${y+s} Z`;}
function cityPath(x,y,s){ // city: house plus tower block
 return `M${x-s*1.5} ${y+s} L${x-s*1.5} ${y-s*0.1} L${x-s*0.4} ${y-s*1.0}`+
        ` L${x+s*0.5} ${y-s*0.1} L${x+s*0.5} ${y-s*0.6} L${x+s*1.5} ${y-s*0.6}`+
        ` L${x+s*1.5} ${y+s} Z`;}

function draw(st,pending){
 svg.innerHTML="";
 svg.appendChild(defs());
 const s=GEO.size;
 const vb=svg.getAttribute("viewBox").split(" ").map(Number);
 svg.appendChild(el("rect",{x:vb[0],y:vb[1],width:vb[2],height:vb[3],fill:"url(#water)"}));
 svg.appendChild(el("rect",{x:vb[0],y:vb[1],width:vb[2],height:vb[3],fill:"url(#waves)"}));
 // coastline: a sand ring under the island, drawn from oversized hexes
 const isleFilter=el("filter",{id:"isleglow",x:"-20%",y:"-20%",width:"140%",height:"140%"});
 isleFilter.appendChild(el("feDropShadow",{dx:0,dy:0,stdDeviation:9,"flood-color":"#e8d9a8","flood-opacity":.35}));
 svg.appendChild(defsWrap(isleFilter));
 GEO.tiles.forEach(t=>svg.appendChild(el("polygon",
  {points:hexPts(t.cx,t.cy,s+8),fill:"#e8d9a8",opacity:.55,filter:"url(#isleglow)"})));
 GEO.tiles.forEach(t=>svg.appendChild(el("polygon",
  {points:hexPts(t.cx,t.cy,s+7),fill:"#d9c48f",opacity:.6})));
 GEO.tiles.forEach(t=>svg.appendChild(el("polygon",
  {points:hexPts(t.cx,t.cy,s+3),fill:"#0e1c22",opacity:.85})));
 GEO.tiles.forEach((t,ti)=>{
  const res=t.resource;
  svg.appendChild(el("polygon",{points:hexPts(t.cx,t.cy,s-1),fill:"url(#g"+res+")",
   stroke:"#0d161b","stroke-width":3.5,"stroke-linejoin":"round","class":"tile"},
   res.toLowerCase()+(t.number?(", rolls on "+t.number):"")));
  svg.appendChild(el("polygon",{points:hexPts(t.cx,t.cy,s-1),fill:"url(#t"+res+")",
   opacity:.5,"pointer-events":"none"}));
  svg.appendChild(el("polygon",{points:hexPts(t.cx,t.cy,s-6),fill:"none",
   stroke:"rgba(255,255,255,.10)","stroke-width":1.5,"pointer-events":"none"}));
  if(t.number){
   svg.appendChild(el("circle",{cx:t.cx,cy:t.cy,r:15,fill:"#f6efdb",
    stroke:"#b8a878","stroke-width":1.5,filter:"url(#drop)"}));
   const hot=(t.number==6||t.number==8);
   const tx=el("text",{x:t.cx,y:t.cy+5,"text-anchor":"middle","font-size":15,
    "font-weight":700,fill:hot?"#b3311f":"#3a3428"});
   tx.textContent=t.number;svg.appendChild(tx);
   const pips=6-Math.abs(7-t.number);
   for(let k=0;k<pips;k++){
    svg.appendChild(el("circle",{cx:t.cx+(k-(pips-1)/2)*5.5,cy:t.cy+19,r:1.9,
     fill:hot?"#b3311f":"#6b5b33"}));}
  }
 });
 GEO.ports.forEach(pt=>{
  const a=GEO.nodes[pt.nodes[0]],b=GEO.nodes[pt.nodes[1]];
  const mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2;
  let dx=mx-CTR[0],dy=my-CTR[1];const L=Math.hypot(dx,dy)||1;dx/=L;dy/=L;
  const lx=mx+dx*27,ly=my+dy*27;
  [a,b].forEach(n=>svg.appendChild(el("circle",{cx:n[0],cy:n[1],r:3.2,
   fill:"#efe2c4",opacity:.9})));
  const w=pt.label.length*6.4+14;
  svg.appendChild(el("rect",{x:lx-w/2,y:ly-9.5,width:w,height:19,rx:9.5,
   fill:"#11262f",opacity:.92,stroke:"#c9a86b","stroke-width":1}));
  const tx=el("text",{x:lx,y:ly+4,"text-anchor":"middle","font-size":10.5,
   fill:"#e8dcc0","font-weight":700});
  tx.textContent=pt.label;svg.appendChild(tx);
 });
 if(st){
  st.roads.forEach(([e,c])=>{const a=GEO.nodes[e[0]],b=GEO.nodes[e[1]];
   svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"#0d161b",
    "stroke-width":11,"stroke-linecap":"round"}));
   svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:PCOLOR[c],
    "stroke-width":6.5,"stroke-linecap":"round"}));
   svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],
    stroke:"rgba(255,255,255,.28)","stroke-width":2,"stroke-linecap":"round"}));});
  st.buildings.forEach(([n,c,t])=>{const p=GEO.nodes[n];
   const d=(t==="CITY")?cityPath(p[0],p[1],7):housePath(p[0],p[1],7);
   svg.appendChild(el("path",{d:d,fill:PCOLOR[c],stroke:"#0d161b","stroke-width":2,
    filter:"url(#drop)"},(t==="CITY"?"City":"Settlement")+" ("+c+")"));});
  // robber: a stubby standing figure (classic wooden-meeple silhouette),
  // with a soft ground shadow and a rim-light edge so it reads as a piece
  // sitting on the tile rather than a flat sticker
  const rt=GEO.tiles[st.robber];
  if(rt){const x=rt.cx+16,y=rt.cy-14,sc=1.15;
   const body=`M${x-6*sc} ${y+11*sc}`+
    ` C${x-7*sc} ${y+2*sc} ${x-5*sc} ${y-1*sc} ${x-3.6*sc} ${y-3*sc}`+
    ` L${x-3.2*sc} ${y-6*sc} C${x-3.2*sc} ${y-9.5*sc} ${x+3.2*sc} ${y-9.5*sc} ${x+3.2*sc} ${y-6*sc}`+
    ` L${x+3.6*sc} ${y-3*sc} C${x+5*sc} ${y-1*sc} ${x+7*sc} ${y+2*sc} ${x+6*sc} ${y+11*sc} Z`;
   svg.appendChild(el("ellipse",{cx:x,cy:y+12*sc,rx:8*sc,ry:2.6*sc,
    fill:"#000",opacity:.4}));
   svg.appendChild(el("path",{d:body,fill:"#1b1f24",stroke:"#e8e8e8",
    "stroke-width":1.1}));
   svg.appendChild(el("path",{d:body,fill:"none",stroke:"rgba(255,255,255,.35)",
    "stroke-width":.8,transform:`translate(-0.8,-0.8)`}));
   svg.appendChild(el("circle",{cx:x,cy:y-9.5*sc,r:3.1*sc,fill:"#1b1f24",
    stroke:"#e8e8e8","stroke-width":1.1},"Robber"));}
 }
 if(!pending) return;
 const tid=pending.turn_id;
 pending.actions.forEach(a=>{
  if(a.kind==="node"){const p=GEO.nodes[a.node];
   const sh=a.city
    ? el("path",{d:cityPath(p[0],p[1],8),fill:"rgba(163,230,53,.18)",stroke:"#a3e635",
       "stroke-width":2,"class":"clicky"},a.label)
    : el("circle",{cx:p[0],cy:p[1],r:9.5,fill:"rgba(163,230,53,.2)",stroke:"#a3e635",
       "stroke-width":2,"class":"clicky"},a.label);
   sh.onclick=()=>act(tid,a.i);svg.appendChild(sh);}
  else if(a.kind==="edge"){const p1=GEO.nodes[a.edge[0]],p2=GEO.nodes[a.edge[1]];
   const ln=el("line",{x1:p1[0],y1:p1[1],x2:p2[0],y2:p2[1],stroke:"#a3e635",
    "stroke-width":7,"stroke-linecap":"round","class":"clicky",opacity:.75},"Build road");
   ln.onclick=()=>act(tid,a.i);svg.appendChild(ln);}
 });
 const byTile={};
 pending.actions.filter(a=>a.kind==="hex").forEach(a=>{(byTile[a.tile]=byTile[a.tile]||[]).push(a);});
 for(const ti in byTile){const t=GEO.tiles[ti];
  const poly=el("polygon",{points:hexPts(t.cx,t.cy,s-8),fill:"rgba(255,90,90,.14)",
   stroke:"#e06060","stroke-width":2.5,"class":"clicky"},"Move robber here");
  poly.onclick=()=>{const opts=byTile[ti];
   if(opts.length===1){act(tid,opts[0].i);return;}
   const v=document.getElementById("victim");
   v.innerHTML="<b>Steal from:</b><br>";
   opts.forEach(o=>{const b=document.createElement("button");
    b.textContent=o.victim||"no one";b.onclick=()=>act(tid,o.i);v.appendChild(b);});
   v.style.display="block";};
  svg.appendChild(poly);}
}

function buildTable(hand){
 const t=document.getElementById("btbl");
 t.innerHTML="<tr><td></td><td style='color:#95a2ab;font-size:11px'>you give</td>"+
  "<td style='color:#95a2ab;font-size:11px'>you want</td></tr>"+
  RES.map(r=>{
   const have=(hand&&hand[r])||0;
   return "<tr><td><span class='chip mini r-"+r+"'>"+r.toLowerCase()+"</span>"+
    "<span style='color:#95a2ab;font-size:11px'> \u00d7"+have+"</span></td>"+
    "<td><div class='stepper'><button onclick=\"bump('give','"+r+"',-1)\">\u2212</button>"+
    "<span class='n' id='g-"+r+"'>"+give[r]+"</span>"+
    "<button onclick=\"bump('give','"+r+"',1)\">+</button></div></td>"+
    "<td><div class='stepper'><button onclick=\"bump('ask','"+r+"',-1)\">\u2212</button>"+
    "<span class='n' id='a-"+r+"'>"+ask[r]+"</span>"+
    "<button onclick=\"bump('ask','"+r+"',1)\">+</button></div></td></tr>";}).join("");
}
function bump(side,r,d){
 const o=(side==="give")?give:ask;
 o[r]=Math.max(0,Math.min(4,o[r]+d));
 document.getElementById((side==="give"?"g-":"a-")+r).textContent=o[r];
 refreshSend();
}
function refreshSend(){
 const g=RES.reduce((s,r)=>s+give[r],0), a=RES.reduce((s,r)=>s+ask[r],0);
 const overlap=RES.some(r=>give[r]>0&&ask[r]>0);
 document.getElementById("send").disabled=!(g>0&&a>0&&!overlap&&window.__canPropose);
}
function sendOffer(){
 post("/propose",{give:give,ask:ask});
 RES.forEach(r=>{give[r]=0;ask[r]=0;});
 document.getElementById("tresult").textContent="Offer sent\u2026";
 refreshSend();
}
function openPicker(title,opts,turn_id){
 document.getElementById("pk-title").textContent=title;
 const box=document.getElementById("pk-opts");box.innerHTML="";
 opts.forEach(o=>{const b=document.createElement("button");b.className="opt";
  b.textContent=o.label;b.onclick=()=>{act(turn_id,o.i);closePicker();};
  box.appendChild(b);});
 document.getElementById("picker").style.display="flex";
}
function closePicker(){document.getElementById("picker").style.display="none";}
document.getElementById("yes").onclick=()=>{post("/trade",{id:offerShown,accept:true});
 document.getElementById("offer").style.display="none";};
document.getElementById("no").onclick=()=>{post("/trade",{id:offerShown,accept:false});
 document.getElementById("offer").style.display="none";};

let lastNotice="";
function paint(d){
 if(d.notice && d.notice!==lastNotice){
  lastNotice=d.notice;
  const n=document.getElementById("notice");
  n.textContent=d.notice;n.style.display="block";
  clearTimeout(window.__nT);
  window.__nT=setTimeout(()=>{n.style.display="none";},6000);
 }
 const key=JSON.stringify([d.log.length,d.pending&&d.pending.turn_id,d.done,
  d.pending_trade&&d.pending_trade.id,d.trade_result,d.can_propose,
  d.trades&&d.trades.length]);
 if(key===lastPaint)return; lastPaint=key;
 draw(d.snapshot,d.pending);
 window.__canPropose=!!d.can_propose;
 const st=document.getElementById("status");
 if(d.done){st.textContent=d.winner?("Game over \u2014 "+d.winner+" wins"):"Game over \u2014 turn limit";
  st.className=d.winner==="BLUE"?"mine":"";}
 else if(d.pending_trade){st.textContent="Trade offered to you";st.className="mine";}
 else if(d.pending){const k=new Set(d.pending.actions.map(a=>a.kind));
  st.textContent=k.has("hex")?"Your turn \u2014 place the robber":
   ((k.has("node")||k.has("edge"))?"Your turn \u2014 pick a glowing spot, or a button":
    "Your turn \u2014 choose an action");
  st.className="mine";}
 else{st.textContent="Bots are thinking\u2026";st.className="";}
 // incoming offer
 const ob=document.getElementById("offer");
 if(d.pending_trade){
  offerShown=d.pending_trade.id;
  const fmt=o=>Object.entries(o).map(([r,n])=>
   "<span class='chip mini r-"+r+"'>"+n+" "+r.toLowerCase()+"</span>").join(" ")||"nothing";
  ob.querySelector(".who").innerHTML="<span style='color:"+
   (PCOLOR[d.pending_trade.proposer]||"#fff")+"'>"+d.pending_trade.proposer+"</span> offers a trade";
  ob.querySelector(".give").innerHTML="You get: "+fmt(d.pending_trade.gives);
  ob.querySelector(".want").innerHTML="You give: "+fmt(d.pending_trade.wants);
  ob.style.display="block";
 } else { ob.style.display="none"; }
 if(d.snapshot){
  document.getElementById("players").innerHTML=d.snapshot.players.map(p=>
   "<div class='p' style='border-left-color:"+PCOLOR[p.color]+"'><span style='color:"+
   PCOLOR[p.color]+"'><b>"+p.color+(p.color==="BLUE"?" (you)":"")+"</b></span>"+
   "<span>"+p.vp+" VP \u00b7 "+p.cards+" cards"+(p.lr?" \u00b7 LR":"")+
   (p.la?" \u00b7 LA":"")+"</span></div>").join("");}
 document.getElementById("hand").innerHTML=Object.entries(d.hand||{})
  .filter(([r,n])=>n>0).map(([r,n])=>"<span class='chip r-"+r+"'>"+r.toLowerCase()+
  " \u00d7 "+n+"</span>").join("")||"<span class='chip' style='background:var(--panel2)'>no cards yet</span>";
 buildTable(d.hand); refreshSend();
 if(d.trade_result)document.getElementById("tresult").textContent=d.trade_result;
 const btns=document.getElementById("buttons");btns.innerHTML="";
 const dc=document.getElementById("devcards");dc.innerHTML="";
 if(d.pending){
  const cards=d.pending.actions.filter(a=>a.kind==="card");
  const bs=d.pending.actions.filter(a=>a.kind==="button");
  cards.forEach(a=>{const info=DEVCARD[a.card]||{icon:"\ud83c\udccf",name:a.card};
   const c=document.createElement("div");
   c.className="devcard dc-"+a.card+(a.locked?" locked":"");
   const opts=a.options;
   const sub=a.locked?"bought this turn \u2014 playable next turn"
    :(opts?(opts.length+" way"+(opts.length===1?"":"s")+" to play"):a.label);
   c.title=a.locked?"Can\u2019t play a card the same turn you bought it"
    :(opts?"Choose how to play":"Play: "+a.label);
   c.innerHTML="<div class='name'>"+info.name.replace("\n","<br>")+"</div>"+
    "<div class='icon'>"+info.icon+"</div><div class='sub'>"+sub+"</div>";
   if(!a.locked){
    c.onclick=()=>{if(opts){openPicker(info.name,opts,d.pending.turn_id);}
     else{act(d.pending.turn_id,a.i);}};
   }
   dc.appendChild(c);});
  if(bs.length){bs.forEach(a=>{const b=document.createElement("button");
   b.textContent=a.label;b.onclick=()=>act(d.pending.turn_id,a.i);btns.appendChild(b);});}
  else if(!cards.length){btns.innerHTML="<div class='none'>All moves are on the board this turn.</div>";}
 }
 const tr=document.getElementById("trades");
 if(tr){tr.innerHTML=(d.trades&&d.trades.length)
  ? d.trades.slice().reverse().map(x=>"<div>"+x+"</div>").join("")
  : "<span style='color:var(--dim)'>none yet</span>";}
 if(d.log.length)document.getElementById("log").innerHTML=
  d.log.slice().reverse().map(x=>"<div>"+x+"</div>").join("");
}
setInterval(()=>fetch(BASE+"/state").then(r=>r.json()).then(paint).catch(()=>{}),600);
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--bind", default="127.0.0.1",
                   help="0.0.0.0 = direct access with a URL token (no tunnel)")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--trade", action="store_true",
                   help="you and the bots trade with each other")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    model, norm = load_agent(args.ckpt)
    bridge = Bridge()
    tcfg = TradeConfig()
    tlog = TradeLog(on_record=bridge.trades.append) if args.trade else None
    bots = build_players(model, norm, "self", deterministic=not args.stochastic,
                         trade=args.trade, trade_log=tlog)
    game = Game(bots, seed=args.seed)

    tiles, node_pos, coord_index, size = board_geometry(game)
    m = game.state.board.map
    node_prod = {n: dict(c) for n, c in m.node_production.items()}
    ports = []
    for port in m.ports_by_id.values():
        trading_nodes = m.port_nodes[port.resource]
        pair = sorted(n for n in port.nodes.values() if n in trading_nodes)
        label = "3:1" if port.resource is None else f"2:1 {port.resource.lower()}"
        ports.append(dict(nodes=pair, label=label))

    xs = [t["cx"] for t in tiles]
    ys = [t["cy"] for t in tiles]
    pad = size * 2.0
    viewbox = (f"{min(xs)-pad:.0f} {min(ys)-pad:.0f} "
               f"{max(xs)-min(xs)+2*pad:.0f} {max(ys)-min(ys)+2*pad:.0f}")
    geo = dict(tiles=tiles,
               nodes={str(k): [round(v[0], 1), round(v[1], 1)]
                      for k, v in node_pos.items()},
               ports=ports, size=size)

    token = "" if args.bind == "127.0.0.1" else secrets.token_urlsafe(8)
    page = (PAGE.replace("__GEO__", json.dumps(geo))
            .replace("__VIEWBOX__", viewbox)
            .replace("__BASE__", f"/{token}" if token else "")
            .replace("__CKPT__", os.path.basename(args.ckpt)))

    human = WebHumanPlayer(Color.BLUE, bridge, node_prod, coord_index,
                           trade_config=tcfg,
                           trade_log=tlog if args.trade else None)
    for i, pl in enumerate(game.state.players):   # seating is shuffled
        if pl.color == Color.BLUE:
            game.state.players[i] = human
            break

    acc = LiveAccumulator(coord_index, bridge)
    threading.Thread(target=lambda: game.play(accumulators=[acc]),
                     daemon=True).start()

    server = ThreadingHTTPServer((args.bind, args.port),
                                 make_handler(bridge, page, token))
    print(f"You are BLUE vs 3x {args.ckpt}"
          f"{' (trading on)' if args.trade else ''}")
    if token:
        print("Direct URL (if this host allows inbound connections):")
        print(f"  http://{socket.getfqdn()}:{args.port}/{token}")
    else:
        print(f"Serving on http://127.0.0.1:{args.port} "
              f"(tunnel local port {args.port} -> 127.0.0.1:{args.port})")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
