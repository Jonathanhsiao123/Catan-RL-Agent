"""Multiplayer rooms: host configures the table, gets a link, friends join.

  python -m catan_rl.lobby --ckpt checkpoints/gen3/ckpt_08000.pt --port 8377
  python -m catan_rl.lobby --ckpt ... --share     # public URL via tunnel

Flow: the first page is the host's room setup (how many humans vs bots; a
Catan table is always 4 seats, matching the trained models). Creating the
room opens the waiting room, which displays the join link with a copy
button. The game starts when every human seat is claimed; remaining seats
are copies of the checkpoint. Spectators can watch without a seat.

--share makes the link work from anywhere: it starts a free public tunnel
(cloudflared if available, else ssh + localhost.run) and shows that URL
instead. Without --share, the printed/displayed URL is only reachable on
your network or through SSH tunnels.
"""

import argparse
import json
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer

from .opponents import NNPlayer
from .play import PAGE, describe_actions
from .watch import ReplayAccumulator, board_geometry, load_agent

SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


# --------------------------------------------------------------------- room
class Room:
    """Holds everything from setup through game over."""

    def __init__(self, ckpt, stochastic, seed, trade=False):
        self.ckpt = ckpt
        self.stochastic = stochastic
        self.seed = seed
        self.trade = trade
        self.trades = []
        self.lock = threading.Lock()
        self.created = False
        # filled by create():
        self.seats = {}
        self.queues = {}
        self.hands = {}
        self.snapshot = None
        self.log = []
        self.pending = None
        self.turn_id = 0
        self.winner = None
        self.done = False
        self.started = False
        self.geo = None
        self.viewbox = None
        self.node_prod = None
        self.coord_index = None

    def create(self, humans):
        with self.lock:
            if self.created:
                return False
            humans = max(1, min(4, humans))
            human_colors = SEAT_COLORS[:humans]
            self.seats = {c: dict(name=None, token=None) for c in human_colors}
            self.queues = {c: queue.Queue() for c in human_colors}
            self.hands = {c: {} for c in human_colors}
            self.created = True

        model, norm = (None, None)
        if humans < 4:
            model, norm = load_agent(self.ckpt)

        game = Game([RandomPlayer(c) for c in SEAT_COLORS], seed=self.seed)
        tiles, node_pos, coord_index, size = board_geometry(game)
        m = game.state.board.map
        self.node_prod = {n: dict(c) for n, c in m.node_production.items()}
        self.coord_index = coord_index
        ports = []
        for port in m.ports_by_id.values():
            trading = m.port_nodes[port.resource]
            pair = sorted(n for n in port.nodes.values() if n in trading)
            label = ("3:1" if port.resource is None
                     else f"2:1 {port.resource.lower()}")
            ports.append(dict(nodes=pair, label=label))
        xs = [t["cx"] for t in tiles]
        ys = [t["cy"] for t in tiles]
        pad = size * 2.0
        self.viewbox = (f"{min(xs)-pad:.0f} {min(ys)-pad:.0f} "
                        f"{max(xs)-min(xs)+2*pad:.0f} {max(ys)-min(ys)+2*pad:.0f}")
        self.geo = dict(
            tiles=tiles,
            nodes={str(k): [round(v[0], 1), round(v[1], 1)]
                   for k, v in node_pos.items()},
            ports=ports, size=size)

        # swap seats into the SAME game whose board we just measured
        for i, pl in enumerate(game.state.players):
            c = pl.color
            if c in self.seats:
                game.state.players[i] = SeatPlayer(c, self)
            else:
                from .trading import TradeLog

                if self.trade and not hasattr(self, "_tlog"):
                    self._tlog = TradeLog(on_record=self.trades.append)
                game.state.players[i] = NNPlayer(
                    c, model, norm, deterministic=not self.stochastic,
                    derived=getattr(model, "uses_derived", False),
                    trade=self.trade,
                    trade_log=getattr(self, "_tlog", None))
        self.game = game

        acc = LiveAccumulator(self.coord_index, self)

        def run():
            import time
            while not self.started:
                time.sleep(0.3)
            self.game.play(accumulators=[acc])

        threading.Thread(target=run, daemon=True).start()
        return True

    def join(self, color_name, player_name):
        with self.lock:
            for c, seat in self.seats.items():
                if c.value == color_name and seat["token"] is None:
                    seat["name"] = player_name or c.value.title()
                    seat["token"] = secrets.token_urlsafe(8)
                    if all(s["token"] for s in self.seats.values()):
                        self.started = True
                    return seat["token"]
        return None

    def seat_of(self, token):
        with self.lock:
            for c, seat in self.seats.items():
                if token and seat["token"] == token:
                    return c
        return None

    def lobby_view(self):
        with self.lock:
            return dict(
                created=self.created, started=self.started,
                seats=[dict(color=c.value, name=s["name"],
                            taken=s["token"] is not None)
                       for c, s in self.seats.items()],
                bots=4 - len(self.seats) if self.created else None)

    def state_view(self, color):
        with self.lock:
            mine = (self.pending is not None and color is not None
                    and self.pending["color"] == color)
            return dict(
                snapshot=self.snapshot, log=self.log[-14:],
                pending=(dict(turn_id=self.pending["turn_id"],
                              actions=self.pending["actions"]) if mine else None),
                trades=self.trades[-6:],
                waiting_for=(None if self.pending is None
                             else self.pending["color"].value),
                hand=self.hands.get(color, {}),
                names={c.value: s["name"] for c, s in self.seats.items()},
                winner=self.winner, done=self.done, started=self.started,
                seat=color.value if color else None)


class SeatPlayer(Player):
    def __init__(self, color, room):
        super().__init__(color)
        self.room = room

    def _hand(self, game):
        idx = game.state.color_to_index[self.color]
        ps = game.state.player_state
        return {r: ps.get(f"P{idx}_{r}_IN_HAND", 0) for r in RESOURCES}

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        r = self.room
        with r.lock:
            r.turn_id += 1
            r.hands[self.color] = self._hand(game)
            r.pending = dict(
                color=self.color, turn_id=r.turn_id,
                actions=describe_actions(playable_actions, r.node_prod,
                                         r.coord_index))
        while True:
            turn_id, idx = r.queues[self.color].get()
            if turn_id == r.turn_id and 0 <= idx < len(playable_actions):
                break
        with r.lock:
            r.pending = None
            r.hands[self.color] = self._hand(game)
        return playable_actions[idx]


class LiveAccumulator(ReplayAccumulator):
    def __init__(self, coord_index, room):
        super().__init__(coord_index)
        self.room = room

    def _publish(self):
        if self.frames:
            self.room.snapshot = self.frames[-1]["state"]
            self.room.log = [f["action"] for f in self.frames]

    def step(self, game_before_action, action):
        super().step(game_before_action, action)
        with self.room.lock:
            self._publish()

    def after(self, game):
        super().after(game)
        with self.room.lock:
            self._publish()
            self.room.winner = self.winner
            self.room.done = True


# ----------------------------------------------------------------- share url
def start_share_tunnel(port, result_holder):
    """Best-effort public URL via cloudflared or ssh+localhost.run."""
    cf = shutil.which("cloudflared") or (
        os.path.expanduser("~/cloudflared")
        if os.path.exists(os.path.expanduser("~/cloudflared")) else None)
    try:
        if cf:
            proc = subprocess.Popen(
                [cf, "tunnel", "--protocol", "http2", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        elif shutil.which("ssh"):
            proc = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 "-R", f"80:localhost:{port}", "nokey@localhost.run"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            pat = re.compile(r"https://[a-z0-9]+\.lhr\.life|https://\S+\.localhost\.run")
        else:
            result_holder["error"] = ("no tunnel tool found: install cloudflared "
                                      "(single binary) or an ssh client")
            return
        for line in proc.stdout:
            mm = pat.search(line)
            if mm:
                result_holder["url"] = mm.group(0)
                print(f"\n  PUBLIC LINK: {mm.group(0)}\n", flush=True)
                break
        proc.stdout.read()  # keep tunnel alive by consuming output
    except Exception as e:  # pragma: no cover
        result_holder["error"] = str(e)


# ---------------------------------------------------------------------- pages
SETUP_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Create Catan room</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&display=swap" rel="stylesheet">
<style>
 body{font-family:system-ui,sans-serif;background:#0f1d23;color:#e9e6df;display:flex;
  flex-direction:column;align-items:center;padding-top:9vh}
 h1{font-family:"Fraunces",Georgia,serif;font-size:34px;margin:0 0 4px}
 .sub{color:#98a4ad;margin-bottom:30px}
 .card{background:#1f262b;border:1px solid rgba(255,255,255,.07);border-radius:14px;
  padding:26px 30px;width:360px;text-align:center}
 .row{display:flex;justify-content:center;gap:8px;margin:14px 0 6px}
 .row button{width:52px;height:52px;border-radius:10px;border:2px solid transparent;
  background:#242c32;color:#e9e6df;font-size:19px;cursor:pointer}
 .row button.sel{border-color:#a3e635;background:#2a4527}
 .hint{color:#98a4ad;font-size:13px;min-height:20px;margin-bottom:16px}
 #go{background:#a3e635;color:#12240b;border:0;border-radius:10px;
  padding:12px 26px;font-size:16px;font-weight:700;cursor:pointer}
 #go:hover{filter:brightness(1.08)}
</style></head><body>
<h1>Catan</h1><div class="sub">create a room &middot; 4 seats at the table &middot; bots: __BOT__</div>
<div class="card">
 <div style="font-size:15px">How many humans?</div>
 <div class="row" id="opts"></div>
 <div class="hint" id="hint"></div>
 <button id="go" onclick="create()">Create room</button>
</div>
<script>
const BASE="__BASE__"; let humans=2;
function render(){
 document.getElementById("opts").innerHTML=[1,2,3,4].map(n=>
  "<button class='"+(n===humans?"sel":"")+"' onclick='pick("+n+")'>"+n+"</button>").join("");
 document.getElementById("hint").textContent =
  humans + " human" + (humans>1?"s":"") + " + " + (4-humans) + " bot" + (4-humans===1?"":"s")
  + (humans===4?" (no bots \u2014 humans only)":"");
}
function pick(n){humans=n;render();}
function create(){
 fetch(BASE+"/create",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({humans:humans})}).then(()=>location=BASE+"/");
}
render();
</script></body></html>
"""

LOBBY_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Catan Lobby</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&display=swap" rel="stylesheet">
<style>
 body{font-family:system-ui,sans-serif;background:#0f1d23;color:#e9e6df;display:flex;
  flex-direction:column;align-items:center;padding-top:7vh}
 h1{font-family:"Fraunces",Georgia,serif;font-size:34px;margin:0 0 4px}
 .sub{color:#98a4ad;margin-bottom:20px}
 #link{display:flex;gap:8px;align-items:center;background:#1f262b;
  border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:10px 14px;margin-bottom:26px}
 #url{font-family:ui-monospace,Consolas,monospace;font-size:13px;color:#a3e635}
 #copy{background:#2e3a42;color:#e9e6df;border:0;border-radius:8px;padding:7px 13px;cursor:pointer}
 #copy:hover{background:#3b4a54}
 input{background:#242c32;border:1px solid rgba(255,255,255,.1);color:#e9e6df;
  border-radius:8px;padding:9px 12px;margin-bottom:20px;width:240px;text-align:center}
 #seats{display:flex;gap:14px;flex-wrap:wrap;justify-content:center}
 .seat{width:170px;background:#1f262b;border:1px solid rgba(255,255,255,.07);
  border-radius:12px;padding:16px;text-align:center;border-top:6px solid}
 .seat b{font-size:17px}
 .seat .who{color:#98a4ad;font-size:13px;min-height:18px;margin:8px 0}
 .seat button{background:#2e3a42;color:#e9e6df;border:0;border-radius:8px;
  padding:8px 14px;cursor:pointer}
 .seat button:hover{background:#3b4a54}
 .bots{color:#98a4ad;font-size:13px;margin-top:18px}
 #spec{margin-top:14px;color:#98a4ad}
 #spec a{color:#a3e635}
</style></head><body>
<h1>Catan</h1><div class="sub">share the link &middot; game starts when every seat fills</div>
<div id="link"><span id="url"></span><button id="copy" onclick="copy()">Copy link</button></div>
<input id="name" placeholder="your name" maxlength="16">
<div id="seats"></div>
<div class="bots" id="bots"></div>
<div id="spec"><a href="__BASE__/play">watch as spectator</a></div>
<script>
const BASE="__BASE__", SHARE="__SHARE__";
const PCOLOR={BLUE:"#3b7dd8",RED:"#d64545",ORANGE:"#e08a2e",WHITE:"#e8e8e8"};
const joinURL = SHARE ? SHARE + BASE + "/" : location.origin + BASE + "/";
document.getElementById("url").textContent = joinURL;
function copy(){navigator.clipboard.writeText(joinURL).then(()=>{
 const b=document.getElementById("copy");b.textContent="Copied!";
 setTimeout(()=>b.textContent="Copy link",1500);});}
function join(c){
 fetch(BASE+"/join",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({color:c,name:document.getElementById("name").value})})
 .then(r=>r.json()).then(d=>{
  if(d.token) location = BASE+"/play?seat="+d.token;
  else alert("Seat taken \u2014 pick another");});
}
function tick(){
 fetch(BASE+"/lobby").then(r=>r.json()).then(d=>{
  document.getElementById("seats").innerHTML = d.seats.map(s=>
   "<div class='seat' style='border-top-color:"+PCOLOR[s.color]+"'>"+
   "<b style='color:"+PCOLOR[s.color]+"'>"+s.color+"</b>"+
   "<div class='who'>"+(s.taken?("\u2713 "+s.name):"open")+"</div>"+
   (s.taken?"":"<button onclick=\\"join('"+s.color+"')\\">Sit here</button>")+
   "</div>").join("");
  document.getElementById("bots").textContent =
   d.bots ? ("+ "+d.bots+" bot"+(d.bots===1?"":"s")+" at the table") : "";});
}
tick(); setInterval(tick, 1200);
</script></body></html>
"""


def adapt_game_page(page):
    page = page.replace(
        'const BASE = "__BASE__";',
        'const BASE = "__BASE__";\n'
        'const SEAT = new URLSearchParams(location.search).get("seat") || "";')
    page = page.replace('fetch(BASE+"/state")',
                        'fetch(BASE+"/state?seat="+SEAT)')
    page = page.replace(
        'body:JSON.stringify({turn_id:turn_id,i:i})',
        'body:JSON.stringify({turn_id:turn_id,i:i,seat:SEAT})')
    page = page.replace(
        'else {st.textContent="Bots are thinking\\u2026";st.className="";}',
        'else if(!d.started){st.textContent="Waiting for players to join\\u2026";st.className="";}\n'
        ' else {const w=d.waiting_for;'
        'st.textContent=(w?((d.names&&d.names[w])||w)+" is thinking\\u2026":"Bots are thinking\\u2026");'
        'st.className="";}')
    page = page.replace(
        '"<b>"+p.color+(p.color==="BLUE"?" (you)":"")+"</b>"',
        '"<b>"+p.color+(p.color===d.seat?" (you)":(d.names&&d.names[p.color]?" \\u00b7 "+d.names[p.color]:""))+"</b>"')
    return page


def make_handler(room, base, share_holder, bot_name):
    prefix = base

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _route(self):
            if prefix and not self.path.startswith(prefix):
                return None, {}
            path = self.path[len(prefix):] if prefix else self.path
            u = urlparse(path)
            return (u.path or "/"), {k: v[0] for k, v in parse_qs(u.query).items()}

        def _send(self, body, ctype):
            body = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route, q = self._route()
            if route is None:
                return self._json({"err": "not found"}, 404)
            if route == "/":
                if not room.created:
                    return self._send(
                        SETUP_PAGE.replace("__BASE__", prefix)
                        .replace("__BOT__", bot_name), "text/html")
                return self._send(
                    LOBBY_PAGE.replace("__BASE__", prefix)
                    .replace("__SHARE__", share_holder.get("url", "")),
                    "text/html")
            if route == "/lobby":
                return self._json(room.lobby_view())
            if route == "/play":
                if not room.created:
                    return self._json({"err": "room not created"}, 409)
                page = (adapt_game_page(PAGE)
                        .replace("__GEO__", json.dumps(room.geo))
                        .replace("__VIEWBOX__", room.viewbox)
                        .replace("__BASE__", prefix)
                        .replace("__CKPT__", bot_name))
                return self._send(page, "text/html")
            if route == "/state":
                return self._json(room.state_view(room.seat_of(q.get("seat"))))
            return self._json({"err": "not found"}, 404)

        def do_POST(self):
            route, _ = self._route()
            if route is None:
                return self._json({"err": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            if route == "/create":
                ok = room.create(int(data.get("humans", 2)))
                return self._json({"ok": ok}, 200 if ok else 409)
            if route == "/join":
                if not room.created:
                    return self._json({"err": "room not created"}, 409)
                tok = room.join(data.get("color", ""),
                                str(data.get("name", "")).strip())
                return self._json({"token": tok} if tok else {"err": "taken"},
                                  200 if tok else 409)
            if route == "/act":
                color = room.seat_of(data.get("seat"))
                if color is None:
                    return self._json({"err": "bad seat"}, 403)
                room.queues[color].put(
                    (int(data.get("turn_id", -1)), int(data.get("i", -1))))
                return self._json({"ok": True})
            return self._json({"err": "not found"}, 404)

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--share", action="store_true",
                   help="publish a public join link via a free tunnel "
                        "(cloudflared or ssh+localhost.run)")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--trade", action="store_true",
                   help="bots trade with each other, priced by their critic")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    room = Room(args.ckpt, args.stochastic, args.seed, trade=args.trade)
    bot_name = os.path.basename(args.ckpt)

    token = ""
    if args.bind != "127.0.0.1" and not args.share:
        token = secrets.token_urlsafe(8)
    base = f"/{token}" if token else ""

    share_holder = {}
    if args.share:
        args.bind = "127.0.0.1"  # tunnel connects locally; no exposure needed
        threading.Thread(target=start_share_tunnel,
                         args=(args.port, share_holder), daemon=True).start()

    server = ThreadingHTTPServer(
        (args.bind, args.port), make_handler(room, base, share_holder, bot_name))
    host = socket.getfqdn() if args.bind != "127.0.0.1" else "localhost"
    print(f"Room server up. Host setup: http://{host}:{args.port}{base}/")
    if args.share:
        print("Waiting for the public link... (appears here and on the lobby page)")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
