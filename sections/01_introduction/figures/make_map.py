#!/usr/bin/env python3
"""
Thesis overview figure, metro-map style.

Only ROUTES (and which column each station sits in) carry meaning. Everything
else is derived:
  * station heights          <- number of lines through the station
  * station vertical order   <- the graph cut at that column (barycentre sort)
  * column vertical shifts   <- least squares, so lines bend as little as possible
  * line bundles             <- true perpendicular offsets of one centreline

  python3 make_map.py tikz > thesis_map.tex     # dissertation
  python3 make_map.py svg  > thesis_map.svg     # PowerPoint
  python3 make_map.py png  [out.png] [--dpi 200] [--tex]   # quick preview
  python3 make_map.py all  [basename]           # writes .tex, .svg and .png

PNG: by default rasterises the SVG backend (needs `pip install cairosvg`). With --tex it
compiles the TikZ backend instead (needs pdflatex + pdftoppm), which shows the LaTeX fonts.
"""
import sys, math
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------- knobs (cm)
COL_DX   = 2.8           # column pitch
ST_W     = 0.55          # station marker (pill) width
ST_PADV  = 0.13          # pill padding above/below the lane band
ST_GROW  = 0.20          # extra pill height per additional lane (1 lane = circle)
LABEL_LINE = 0.32        # label line height (cm) for layout clearance
LABEL_CHAR = 0.15        # label width per character (cm) for layout clearance
LABEL_GAP  = 0.12        # space between pill and its label
GAP      = 0.35          # vertical gap between stacked items in a column cut
LW       = 0.10          # line width
LANE     = LW            # lane pitch == line width: lines touch
CORNER   = 0.18          # bend radius of route corners
RUN_IN   = 0.30          # straight run before a station after a 45/vertical/45 drop
ST_RX    = 0.27          # pill corner radius (== half width gives a true pill)
SHOW_LEGEND = True

# ---------------------------------------------------------------- stations: id -> (column, preferred order top->bottom within column, label)
STATIONS = {   # id -> (column, semantic level 0=top 1=mid 2=bottom, label)
    "src": (0, 1, ["source", "images"]),
    "seg": (1, 0, ["segmentations"]),
    "hcf": (1, 2, ["hand-crafted", "features"]),
    "lat": (2, 0, ["latent codes /", "encodings"]),
    "pc":  (2, 2, ["point clouds"]),
    "smf": (3, 0, ["shape model", "fitting"]),
    "reg": (3, 1, ["registration"]),
    "io":  (4, 1, ["instance", "optimisation"]),
    "pop": (5, 0, ["population", "model"]),
    "def": (5, 1, ["deformation", "field"]),
    "rec": (5, 2, ["reconstruction"]),
}
LEVEL_PULL = 10.0   # stations are ordered by level; pass-through bundles slot between them by barycentre

PHASES = [("Input", [0]), ("Shape representation", [1, 2]), ("Estimation", [3, 4]), ("Result", [5])]
SHOW_PHASES = True
BAND_GREY = 4            # percent grey of the shaded phase bands

# ---------------------------------------------------------------- routing table (the part that matters)
ROUTES = [
    (3, ["src", "seg", "lat", "reg", "io", "def"]),
    (4, ["src", "seg", "lat", "smf", "pop"]),
    (5, ["src", "reg", "pop"]),
    (6, ["src", "hcf", "reg", "io", "def"]),
    (6, ["src", "seg", "reg"]),
    (7, ["src", "seg", "pc", "reg", "io", "rec"]),
    (7, ["src", "hcf", "pc"]),
]
# COLOURS = {3: "0019A8", 4: "9B0056", 5: "00782A", 6: "EF7B10", 7: "E32017"}   # transit palette
CH_TITLES = {   # legend text: the short chapter headings from main.tex (full title where none is set)
    3: "Generalised 3D Medical Image Registration with Learned Shape Encodings",
    4: "Robust Statistical Shape Modelling with Implicit Neural Representations",
    5: "SINA: Sharp Implicit Neural Atlases",
    6: "convexAdam: Self-Configuring 3D Registration",
    7: "Point-Cloud-Based 3D Freehand Ultrasound",
}
LG_ROW = 0.46            # legend row pitch (cm)
BADGE_W   = 1.50         # legend badge ("Chapter N" in a rounded bar of the line colour)
BADGE_H   = 0.30
DARK_TEXT = {5}          # chapters whose badge needs dark text (light line colour)
COLOURS = {3: "D62828", 4: "F77F00", 5: "E9C000", 6: "2A9D3F", 7: "1F5FBF"}      # rainbow, Ch.3 -> Ch.7
STATION_STROKE = "4A4E53"

# ================================================================ graph
NCOL = max(c for c, *_ in STATIONS.values()) + 1
COL_X = [c * COL_DX for c in range(NCOL)]
col_of = {s: STATIONS[s][0] for s in STATIONS}
edges = [(ch, a, b) for ch, route in ROUTES for a, b in zip(route, route[1:])]

def n_lines(sid):
    return max(sum(1 for _, a, _ in edges if a == sid), sum(1 for _, _, b in edges if b == sid))

def st_h(sid):
    n = n_lines(sid)
    return max(ST_W + (n - 1) * ST_GROW, n * LANE + 2 * ST_PADV)   # 1 lane: circle; grows per extra lane

def marker_kind(sid):
    """origin (no incoming lines) / terminus (no outgoing) / station"""
    if not any(b == sid for _, _, b in edges): return "origin"
    if not any(a == sid for _, a, _ in edges): return "terminus"
    return "station"
BAR_W = 0.16             # terminus bar thickness

# bundles: edges that share the same (A, B) run together; order inside by chapter
bundle_of = {}
bundles = defaultdict(list)
for i, (ch, a, b) in enumerate(edges):
    bundles[(a, b)].append(i)
for k, ids in bundles.items():
    ids.sort(key=lambda i: edges[i][0])
    for r, i in enumerate(ids):
        bundle_of[i] = (k, r, len(ids))

def rel_off(i):                      # perpendicular offset of edge i inside its bundle
    _, r, n = bundle_of[i]
    return ((n - 1) / 2 - r) * LANE

# ================================================================ column cuts -> vertical layout
# items in a column cut: ("st", sid) or ("bd", (a,b))  [a bundle passing through]
def cut_items(c):
    items = [("st", s) for s in STATIONS if col_of[s] == c]
    for (a, b) in bundles:
        if col_of[a] < c < col_of[b]:
            items.append(("bd", (a, b)))
    return items

def item_h(it):
    return st_h(it[1]) if it[0] == "st" else len(bundles[it[1]]) * LANE

# ---------------------------------------------------------------- cut order + ports (columns left to right)
ypos = {}            # (col, item) -> y (provisional, used for barycentres)
order = {}           # col -> items top to bottom
port_in, port_out = {}, {}

def edge_y_at(i, c):
    ch, a, b = edges[i]
    if c == col_of[a]:
        return ypos[(c, ("st", a))] + port_out[i]
    if c == col_of[b]:
        return ypos[(c, ("st", b))] + port_in[i]
    return ypos[(c, ("bd", (a, b)))] + rel_off(i)

def place_bundles(keys, chapters_of):
    """Contiguous slots for an ordered list of bundles; a bundle that shares a chapter
    with the previous one is aligned so that chapter keeps its lane (early union)."""
    ports, top, prev = {}, 0.0, None
    for k in keys:
        n = len(bundles[k])
        centre = top - (n - 1) / 2 * LANE
        if prev is not None:
            shared = [ch for ch in chapters_of(k) if ch in chapters_of(prev)]
            if shared:
                ch = shared[0]
                i_prev = next(i for i in bundles[prev] if edges[i][0] == ch)
                i_new = next(i for i in bundles[k] if edges[i][0] == ch)
                centre = ports[i_prev] - rel_off(i_new)
        for i in bundles[k]:
            ports[i] = centre + rel_off(i)
        top = min(ports[i] for i in bundles[k]) - LANE
        prev = k
    if not ports:
        return ports
    mid = (max(ports.values()) + min(ports.values())) / 2      # recentre on the station
    return {i: p - mid for i, p in ports.items()}

chapters = lambda k: [edges[i][0] for i in bundles[k]]

def continue_lanes(sid, dep):
    """Departure lanes at a through station: one contiguous band (bundles in
    destination order, exactly LANE apart). Its offset is chosen so that as many
    continuing lines as possible keep their arrival lane (weighted median;
    straight-on lines count double)."""
    in_lane = {}
    for i in port_in:
        if edges[i][2] == sid:
            in_lane.setdefault(edges[i][0], []).append(port_in[i])
    # relative position of every departing edge inside the band (top = 0)
    rel, k = {}, 0
    for key in dep:
        for i in bundles[key]:
            rel[i] = -k * LANE; k += 1
    cands = []
    for i in rel:
        ch = edges[i][0]
        if ch in in_lane:
            w = 2.0 if STATIONS[edges[i][2]][1] == STATIONS[sid][1] else 1.0
            for L in in_lane[ch]:
                cands.append((L - rel[i], w))
    if cands:                                     # weighted median of desired band tops
        cands.sort()
        tot = sum(w for _, w in cands); acc = 0.0
        top = cands[-1][0]
        for v, w in cands:
            acc += w
            if acc >= tot / 2: top = v; break
    else:
        top = (k - 1) / 2 * LANE
    return {i: top + rel[i] for i in rel}

for c in range(NCOL):
    items = cut_items(c)
    def bary(it):
        if c == 0:
            return (STATIONS[it[1]][1],)
        if it[0] == "st":
            ins = [i for i, (_, a, b) in enumerate(edges) if b == it[1]]
            return (-np.mean([edge_y_at(i, c - 1) for i in ins]) + (STATIONS[it[1]][1] - 1) * LEVEL_PULL,
                    STATIONS[it[1]][1])
        return (-np.mean([edge_y_at(i, c - 1) for i in bundles[it[1]]]), 1)
    items.sort(key=bary)
    order[c] = items
    for it in items:                       # provisional y: packed stack centred on 0
        pass
    total = sum(item_h(it) for it in items) + GAP * (len(items) - 1)
    y = total / 2
    for it in items:
        h = item_h(it); ypos[(c, it)] = y - h / 2; y -= h + GAP
    for it in items:
        if it[0] != "st":
            continue
        sid = it[1]
        arr = [(a, b) for (a, b) in bundles if b == sid]
        arr.sort(key=lambda k: -np.mean([edge_y_at(i, c - 1) for i in bundles[k]]))
        port_in.update(place_bundles(arr, chapters))
        dep = [(a, b) for (a, b) in bundles if a == sid]
        dep.sort(key=lambda k: (STATIONS[k[1]][1], edges[bundles[k][0]][0]))
        if not any(edges[i][2] == sid for i in range(len(edges))):
            port_out.update(place_bundles(dep, chapters))              # origin: plain fan-out
        else:
            port_out.update(continue_lanes(sid, dep))                  # through lines keep their lane
        # marker is centred on the union of arrival and departure lanes
        lanes = [port_in[i] for i in port_in if edges[i][2] == sid] + [port_out[i] for i in port_out if edges[i][1] == sid]
        if lanes:
            mid = (max(lanes) + min(lanes)) / 2
            for i in list(port_in):
                if edges[i][2] == sid: port_in[i] -= mid
            for i in list(port_out):
                if edges[i][1] == sid: port_out[i] -= mid

# station height from the number of distinct lanes actually used
def st_h(sid):
    lanes_in = {round(port_in[i], 4) for i in port_in if edges[i][2] == sid}
    lanes_out = {round(port_out[i], 4) for i in port_out if edges[i][1] == sid}
    n = max(len(lanes_in), len(lanes_out), 1)
    allv = lanes_in | lanes_out
    span = (max(allv) - min(allv) + LANE) if allv else LANE
    return max(ST_W + (n - 1) * ST_GROW, span + 2 * ST_PADV)

# labels sit outside the pill: above for top-level stations, below otherwise
def label_side(sid):
    return "above" if STATIONS[sid][1] == 0 else "below"
def label_h(sid):
    return len(STATIONS[sid][2]) * LABEL_LINE + LABEL_GAP
def label_w(sid):
    return max(len(t.lstrip("*")) for t in STATIONS[sid][2]) * LABEL_CHAR
def extents(it):
    """(up, down) clearance of an item in its column cut, label included for mid stations."""
    if it[0] == "bd":
        h = len(bundles[it[1]]) * LANE; return h / 2, h / 2
    sid = it[1]; h = st_h(sid) / 2
    if STATIONS[sid][1] == 1 and marker_kind(sid) != "terminus":
        return h, h + label_h(sid)
    return h, h

def item_h(it):
    return st_h(it[1]) if it[0] == "st" else len(bundles[it[1]]) * LANE

# ---------------------------------------------------------------- layout: one mixed-integer program
# Unknowns: y of every free station and pass-through bundle, the height of every station
# (>= its line-count height), and a binary "bends?" per hop. Cost: bend heights (L1) +
# fixed cost per bend + small cost per cm of extra station height. Lanes are always centred.
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

FRAME_COST  = 0.15   # cost per cm of frame height above the minimum (the frame may grow to avoid bends)
BEND_COST   = 1.5    # fixed cost per bend (cm-equivalents)
GROW_COST   = 0.05   # (unused with pill markers: heights are fixed)
FILL_MIDS   = False  # pills do not grow
FLOAT_MAX   = 0.6    # top/bottom markers may sit up to this far inside the frame (labels stay on their row)
BIG         = 30.0

def solve_layout():
    global H_FRAME
    base_h = {sid: st_h(sid) for sid in STATIONS}
    H_MIN = max(sum(sum(extents(it)) for it in order[c]) + GAP * (len(order[c]) - 1) for c in range(NCOL))

    # ---- variables
    var = {}                                    # name -> column index
    def v(name):
        if name not in var: var[name] = len(var)
        return var[name]
    v(("H",))                                   # frame height
    for sid in STATIONS:
        v(("h", sid))
    for c in range(NCOL):
        for it in order[c]:
            if it[0] == "bd": v(("bd", it[1]))
            else: v(("y", it[1]))
    nx = len(var)

    # linear expressions: dict var->coef, plus const
    def y_expr(c, it):
        if it[0] == "bd":
            return {var[("bd", it[1])]: 1.0}, 0.0
        sid, lvl = it[1], STATIONS[it[1]][1]
        return {var[("y", sid)]: 1.0}, 0.0
    def h_expr(it):
        if it[0] == "bd": return {}, len(bundles[it[1]]) * LANE
        return {var[("h", it[1])]: 1.0}, 0.0
    def lane_expr(i, c):
        ch, a, b = edges[i]
        if c == col_of[a]:   e, k = y_expr(c, ("st", a)), port_out[i]
        elif c == col_of[b]: e, k = y_expr(c, ("st", b)), port_in[i]
        else:                e, k = y_expr(c, ("bd", (a, b))), rel_off(i)
        return e[0], e[1] + k

    # ---- hops:  lane(c) - lane(c+1) = r_j ;  |r_j| <= t_j ;  t_j <= BIG z_j
    hop_rows, hop_const, hop_w = [], [], []
    for i, (ch, a, b) in enumerate(edges):
        for c in range(col_of[a], col_of[b]):
            (e1, k1), (e2, k2) = lane_expr(i, c), lane_expr(i, c + 1)
            row = np.zeros(nx)
            for j, cf in e1.items(): row[j] += cf
            for j, cf in e2.items(): row[j] -= cf
            hop_rows.append(row); hop_const.append(k1 - k2); hop_w.append(1.0 / len(bundles[(a, b)]))
    m = len(hop_rows)
    A_ = np.array(hop_rows); b_ = -np.array(hop_const)     # A x - b = residual
    NV = nx + 2 * m
    def full(row_x, t=None, z=None):
        r = np.zeros(NV); r[:nx] = row_x
        if t is not None: r[nx + t[0]] = t[1]
        if z is not None: r[nx + m + z[0]] = z[1]
        return r
    rows, ub = [], []
    for j in range(m):
        rows.append(full(A_[j], t=(j, -1))); ub.append(b_[j])          #  A x - t <= b
        rows.append(full(-A_[j], t=(j, -1))); ub.append(-b_[j])        # -A x - t <= -b
        rows.append(full(np.zeros(nx), t=(j, 1), z=(j, -BIG))); ub.append(0)   # t <= BIG z
    # ---- ordering + gaps inside each cut:  y_v + h_v/2 + gap <= y_u - h_u/2
    for c in range(NCOL):
        its = order[c]
        for k in range(1, len(its)):
            u, w_ = its[k-1], its[k]
            g = 0.0 if (u[0] == "bd" and w_[0] == "bd" and u[1][1] == w_[1][1]) else GAP
            row = np.zeros(nx); const = 0.0
            for e, cf in ((y_expr(c, w_), 1), (y_expr(c, u), -1)):
                for j, x in e[0].items(): row[j] += cf * x
                const += cf * e[1]
            const += extents(u)[1] + extents(w_)[0]        # fixed heights (pills + labels)
            rows.append(full(row)); ub.append(-g - const)
    # ---- frame: stations must stay inside it   (y + h/2 - H/2 <= 0 ; -y + h/2 - H/2 <= 0)
    #      and top/bottom stations must stay within FLOAT_MAX of their frame edge
    for c in range(NCOL):
        for it in order[c]:
            if it[0] != "st": continue
            lvl = STATIONS[it[1]][1]
            hh = st_h(it[1]) / 2                       # frame bounds the markers; mid-station labels may hang below it
            for sgn, ext in ((1, hh), (-1, hh)):
                row = np.zeros(nx); const = ext
                for e, cf in ((y_expr(c, it), sgn),):
                    for j, x in e[0].items(): row[j] += cf * x
                    const += cf * e[1]
                row[var[("H",)]] -= 0.5
                rows.append(full(row)); ub.append(-const)
                if (lvl == 0 and sgn == 1) or (lvl == 2 and sgn == -1):     # -(y·sgn + ext) + H/2 <= FLOAT_MAX
                    rows.append(full(-row)); ub.append(FLOAT_MAX + const)
    # ---- bounds and objective
    lb = np.full(NV, -np.inf); hb = np.full(NV, np.inf)
    lb[var[("H",)]] = H_MIN
    for sid in STATIONS:
        j = var[("h", sid)]; lb[j] = base_h[sid]
        lone_mid = STATIONS[sid][1] == 1 and not any(it[0] == "bd" for it in order[col_of[sid]]) and len(order[col_of[sid]]) > 1
        hb[j] = base_h[sid]                                              # pills: fixed height
    lb[nx:nx + m] = 0; lb[nx + m:] = 0; hb[nx + m:] = 1
    cost = np.zeros(NV)
    cost[nx:nx + m] = hop_w; cost[nx + m:] = BEND_COST * np.array(hop_w)
    cost[var[("H",)]] = FRAME_COST
    for sid in STATIONS:                       # gentle pull of top/bottom markers toward their label row
        if ("y", sid) in var and STATIONS[sid][1] in (0, 2):
            cost[var[("y", sid)]] = 0.0
    for sid in STATIONS:
        cost[var[("h", sid)]] = GROW_COST if not (FILL_MIDS and STATIONS[sid][1] == 1) else -GROW_COST
    integrality = np.zeros(NV); integrality[nx + m:] = 1
    res = milp(cost, constraints=[LinearConstraint(csr_matrix(np.array(rows)), -np.inf, np.array(ub))],
               bounds=Bounds(lb, hb), integrality=integrality)
    assert res.status == 0, res.message
    x = res.x[:nx]
    H_FRAME = x[var[("H",)]]; Y_TOP, Y_BOT = H_FRAME / 2, -H_FRAME / 2

    heights = {sid: x[var[("h", sid)]] for sid in STATIONS}
    Yc = {}
    for c in range(NCOL):
        for it in order[c]:
            e, k = y_expr(c, it)
            Yc[(c, it)] = sum(x[j] * cf for j, cf in e.items()) + k
    band = {sid: 0.0 for sid in STATIONS}
    def lane_terms(i, c):
        ch, a, b = edges[i]
        if c == col_of[a]:   return (c, ("st", a)), port_out[i]
        if c == col_of[b]:   return (c, ("st", b)), port_in[i]
        return (c, ("bd", (a, b))), rel_off(i)
    body = {s: (COL_X[col_of[s]], Yc[(col_of[s], ("st", s))], ST_W, heights[s]) for s in STATIONS}
    return dict(Yc=Yc, band=band, body=body, order=order, Y_TOP=Y_TOP, Y_BOT=Y_BOT, heights=heights,
                lane_terms=lane_terms, A_=A_, b_=b_, sol=x)

L1 = solve_layout()
globals().update(L1)
_h_final = L1["heights"]
def st_h(sid):
    return _h_final[sid]

# ================================================================ routing
def hop(p, q, x_limit=None, a=None, b=None):
    """Octilinear hop from waypoint p to q. The 45-degree segment starts at the
    station edge; it may run past the next column up to x_limit if given.
    a/b: station ids at the ends (None for pass-through waypoints), used to keep
    downward diagonals clear of a label hanging below a mid station."""
    (x1, y1), (x2, y2) = p, q
    xs, xe = x1 + ST_W / 2, x2 - ST_W / 2
    dy = y2 - y1
    if a is not None and STATIONS[a][1] == 1 and dy < -0.02:      # leaving downward under its own label
        xs = max(xs, x1 + label_w(a) / 2 + 0.1)
    if b is not None and STATIONS[b][1] == 1 and dy > 0.02:       # arriving from below under the target's label
        xe = min(xe, x2 - label_w(b) / 2 - 0.1)
    if abs(dy) < 0.02:
        return [(xs, y1), (xe, y1)]
    if abs(dy) <= 2 * LANE:                    # lane-sized shift: do it under a station marker, not in the open
        if a is not None:
            return [(x1, y1), (x1 + abs(dy), y2), (xe, y2)]
        if b is not None:
            return [(xs, y1), (x2 - abs(dy), y1), (x2, y2)]
    avail = (x_limit if x_limit is not None else xe) - xs
    sg = 1 if dy > 0 else -1
    if abs(dy) <= avail - CORNER:
        return [(xs, y1), (xs + abs(dy), y2), (xe, y2)]   # bend straight away (early union)
    avail = xe - xs - RUN_IN
    d = max(0.0, min(avail, abs(dy) - (2 * CORNER + 0.15)) / 2)
    return [(xs, y1), (xs + d, y1 + sg * d), (xs + d, y2 - sg * d), (xs + 2 * d, y2), (xe, y2)]

def hits_station(seg_a, seg_b, cols, margin=0.15):
    """Does the segment a-b cross any station box in the given columns?"""
    for sid, (x, cy, w, h) in body.items():
        if col_of[sid] not in cols:
            continue
        x0, x1, y0, y1 = x - w/2 - margin, x + w/2 + margin, cy - h/2 - margin, cy + h/2 + margin
        # sample the segment
        for t in np.linspace(0, 1, 40):
            px, py = seg_a[0] + t * (seg_b[0] - seg_a[0]), seg_a[1] + t * (seg_b[1] - seg_a[1])
            if x0 < px < x1 and y0 < py < y1:
                return True
    return False

def centreline(key):
    """Bundle centreline through the column waypoints, as a polyline. A diagonal
    may run through pass-through columns if it clears every station there."""
    a, b = key
    ids = bundles[key]
    ca, cb = col_of[a], col_of[b]
    ways = []
    for c in range(ca, cb + 1):
        ys = [Yc[lane_terms(i, c)[0]] + lane_terms(i, c)[1] - rel_off(i) for i in ids]
        ways.append((COL_X[c], float(np.mean(ys))))
    pts = [ways[0]]
    k = 0
    while k < len(ways) - 1:
        p = ways[k]
        # furthest waypoint reachable on the same level as ways[k+1] (bundle runs level between stations)
        j = k + 1
        while j + 1 < len(ways) and abs(ways[j + 1][1] - ways[k + 1][1]) < 1e-6:
            j += 1
        q = ways[j]
        seg = hop(p, q, x_limit=q[0] - ST_W / 2, a=a if k == 0 else None, b=b if j == len(ways) - 1 else None)
        # diagonal must clear stations in the columns it crosses
        cols = set(range(ca + k + 1, ca + j))
        if len(seg) == 3 and cols and hits_station(seg[0], seg[1], cols):
            seg = hop(p, ways[k + 1], a=a if k == 0 else None, b=b if k + 1 == len(ways) - 1 else None)
            j = k + 1
            pts += seg
            k = j
            continue
        if len(seg) == 5 and j > k + 1:                # long vertical form: keep per-column routing
            seg = hop(p, ways[k + 1], a=a if k == 0 else None, b=b if k + 1 == len(ways) - 1 else None); j = k + 1
        pts += seg
        k = j
    pts.append(ways[-1])
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out

def offset_polyline(pts, d):
    """True parallel offset (mitred) of a polyline by distance d (left of travel)."""
    if abs(d) < 1e-9:
        return list(pts)
    segs = []
    for p, q in zip(pts, pts[1:]):
        ux, uy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(ux, uy); ux, uy = ux / L, uy / L
        nx, ny = -uy * d, ux * d
        segs.append(((p[0] + nx, p[1] + ny), (q[0] + nx, q[1] + ny)))
    out = [segs[0][0]]
    for (p1, p2), (q1, q2) in zip(segs, segs[1:]):
        # intersection of lines p1p2 and q1q2
        x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            out.append(p2)
        else:
            px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
            py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
            out.append((px, py))
    out.append(segs[-1][1])
    return out

def corner_radii(cl, d):
    """Concentric bends: a line offset d to the left of travel gets radius CORNER - d
    on left turns (inside) and CORNER + d on right turns (outside)."""
    rs = []
    for k in range(1, len(cl) - 1):
        ux, uy = cl[k][0] - cl[k-1][0], cl[k][1] - cl[k-1][1]
        vx, vy = cl[k+1][0] - cl[k][0], cl[k+1][1] - cl[k][1]
        turn = 1 if ux * vy - uy * vx > 0 else -1
        rs.append(max(0.01, CORNER - turn * d))
    return rs

paths, radii = {}, {}
for key, ids in bundles.items():
    cl = centreline(key)
    for i in ids:
        paths[i] = offset_polyline(cl, rel_off(i))
        radii[i] = corner_radii(cl, rel_off(i))

draw_order = sorted(range(len(edges)), key=lambda i: -edges[i][0])   # Ch.3 on top

# ================================================================ shared frame geometry
def content_extent():
    nt = [s for s in STATIONS if marker_kind(s) != "terminus"]
    ytop = max([cy + h/2 + label_h(s) for s, (x, cy, w, h) in body.items() if STATIONS[s][1] == 0 and s in nt] +
               [cy + h/2 + 0.15 for s, (x, cy, w, h) in body.items()])
    ybot = min([cy - h/2 - label_h(s) for s, (x, cy, w, h) in body.items() if STATIONS[s][1] != 0 and s in nt] +
               [cy - h/2 - 0.15 for s, (x, cy, w, h) in body.items()])
    return ytop, ybot
def band_boxes():
    """(name, x0, x1) of each phase band; boundaries at column midpoints."""
    half = COL_DX / 2
    out = []
    for name, cols in PHASES:
        x1 = COL_X[cols[-1]] + half
        if cols[-1] == NCOL - 1:
            x1 = max(x1, max(COL_X[NCOL-1] + label_w(s) + 0.35 for s in STATIONS if marker_kind(s) == "terminus"))
        out.append((name, COL_X[cols[0]] - half, x1))
    return out

# ================================================================ TikZ backend
def badge_inline(ch):
    """The badge as an inline picture, for use inside the legend table."""
    node = badge_node(ch, "(0,0)").replace("\\node[", "\\node(b)[", 1)
    return r"\tikz[baseline=(b.base)]" + node

def badge_node(ch, at):
    """Rounded bar in the line colour with "Chapter N" inside; fixed width so the titles align."""
    txt = "black!85" if ch in DARK_TEXT else "white"
    return (rf"\node[anchor=west, fill=c{ch}, rounded corners={BADGE_H/2:.3f}cm, inner sep=0pt, "
            rf"minimum width={BADGE_W:.2f}cm, minimum height={BADGE_H:.2f}cm, "
            rf"text={txt}, font=\scriptsize\sffamily\bfseries] at {at} {{Chapter~{ch}}};")

def emit_tikz():
    L = [r"\documentclass[tikz,border=6pt]{standalone}",
         r"\usepackage{lmodern}\usepackage[T1]{fontenc}"]
    for ch, c in COLOURS.items():
        L.append(rf"\definecolor{{c{ch}}}{{HTML}}{{{c}}}")
    L.append(rf"\definecolor{{stB}}{{HTML}}{{{STATION_STROKE}}}")
    L.append(r"""\begin{document}
\begin{tikzpicture}[
  stn/.style={draw=stB, fill=white, line width=0.6pt, rounded corners=%.2fcm},
  rt/.style={line width=%.2fcm, line cap=butt, line join=round},
  lbl/.style={font=\scriptsize, align=center, text=black!80, inner sep=0pt}]""" % (ST_RX, LW))
    ytop_c, ybot_c = content_extent()
    if SHOW_PHASES:
        y0, y1 = ybot_c - 0.12, ytop_c + 0.15
        L.append("% ---- phase bands ----")
        for n, (name, x0, x1) in enumerate(band_boxes()):
            if n % 2 == 1:
                L.append(rf"\fill[black!{BAND_GREY}] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
            glyph = ""
            L.append(rf"\node[font=\scriptsize\scshape, text=black!60, anchor=south] at ({(x0+x1)/2:.3f},{y1 + 0.08:.3f}) {{{glyph}{name}}};")
    L.append("% ---- routes (generated; do not hand-edit) ----")
    for i in draw_order:
        ch, a, b = edges[i]
        P = paths[i]
        parts = [f"({P[0][0]:.3f},{P[0][1]:.3f})"]
        for k in range(1, len(P)):
            r = CORNER
            if k < len(P) - 1:      # concentric radius, never more than half the shorter neighbour
                l1 = math.dist(P[k-1], P[k]); l2 = math.dist(P[k], P[k+1])
                r = min(radii[i][k-1], l1 / 2, l2 / 2)
            parts.append(f"[rounded corners={r:.3f}cm] -- ({P[k][0]:.3f},{P[k][1]:.3f})")
        L.append(rf"\draw[rt, c{ch}] {' '.join(parts)};   % Ch.{ch}: {a} -> {b}")
    L.append("% ---- stations (labels inside; swap for \\includegraphics later) ----")
    for sid, (x, cy, w, h) in body.items():
        rx = min(ST_RX, h / 2 - 0.005, w / 2 - 0.005)
        kind = marker_kind(sid)
        if kind == "terminus":
            L.append(rf"\draw[stB, line width={BAR_W:.2f}cm, line cap=round] ({x:.3f},{cy-h/2+0.05:.3f}) -- ({x:.3f},{cy+h/2-0.05:.3f});")
        elif kind == "origin":
            L.append(rf"\draw[stn, fill=stB, rounded corners={rx:.3f}cm] ({x-w/2:.3f},{cy-h/2:.3f}) rectangle ({x+w/2:.3f},{cy+h/2:.3f});")
        else:
            L.append(rf"\draw[stn, rounded corners={rx:.3f}cm] ({x-w/2:.3f},{cy-h/2:.3f}) rectangle ({x+w/2:.3f},{cy+h/2:.3f});")
        txt = r"\\".join((r"{\itshape " + s[1:] + "}") if s.startswith("*") else s for s in STATIONS[sid][2])
        lvl = STATIONS[sid][1]
        if marker_kind(sid) == "terminus":
            L.append(rf"\node[lbl, anchor=west, align=left] at ({x + BAR_W/2 + 0.12:.3f},{cy:.3f}) {{{txt}}};")
        elif lvl == 0:
            L.append(rf"\node[lbl, anchor=south] at ({x:.3f},{cy + h/2 + LABEL_GAP:.3f}) {{{txt}}};")
        elif lvl == 2:
            L.append(rf"\node[lbl, anchor=north] at ({x:.3f},{cy - h/2 - LABEL_GAP:.3f}) {{{txt}}};")
        else:
            L.append(rf"\node[lbl, anchor=north] at ({x:.3f},{cy - h/2 - LABEL_GAP:.3f}) {{{txt}}};")
    if SHOW_LEGEND:
        ylg = content_extent()[1] - (0.25 if SHOW_PHASES else 0) - 0.5
        bb = band_boxes()
        xc = (bb[0][1] + bb[-1][2]) / 2                                    # horizontal centre of the map
        rows = [rf"  {badge_inline(ch)} & {CH_TITLES[ch]}" for ch in sorted(COLOURS)]   # one row per chapter: badge + title
        L.append(rf"\node[anchor=north, inner sep=0pt, font=\scriptsize, text=black!80] at ({xc:.3f},{ylg:.3f}) {{%")
        L.append(rf" \begin{{tabular}}{{@{{}}l@{{\hspace{{0.18cm}}}}l@{{}}}}")
        L.append((rf" \\[{LG_ROW - BADGE_H:.2f}cm]" + "\n").join(rows))
        L.append(r" \end{tabular}};")
    L.append(r"\end{tikzpicture}" "\n" r"\end{document}")
    return "\n".join(L)

# ================================================================ SVG backend
def rounded_path(pts, rs):
    def towards(p, q, d):
        L = math.hypot(q[0]-p[0], q[1]-p[1]); d = min(d, L/2)
        return (p[0] + (q[0]-p[0]) * d / L, p[1] + (q[1]-p[1]) * d / L)
    d = [f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for k in range(1, len(pts) - 1):
        r = rs[k-1]
        a = towards(pts[k], pts[k-1], r); b = towards(pts[k], pts[k+1], r)
        d.append(f"L {a[0]:.1f},{a[1]:.1f} Q {pts[k][0]:.1f},{pts[k][1]:.1f} {b[0]:.1f},{b[1]:.1f}")
    d.append(f"L {pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)

def emit_svg():
    S = 37.8
    x0 = min(x - max(w/2, label_w(s)/2) for s, (x, cy, w, h) in body.items()) - 0.3
    x1 = max(x + (label_w(s) + 0.3 if marker_kind(s) == "terminus" else max(w/2, label_w(s)/2)) for s, (x, cy, w, h) in body.items()) + 0.3
    ytop_c, ybot_c = content_extent()
    ytop = ytop_c + (1.0 if SHOW_PHASES else 0.3)
    ybot = ybot_c - (0.25 if SHOW_PHASES else 0) - ((0.9 + len(COLOURS) * LG_ROW) if SHOW_LEGEND else 0.3)
    W, H = (x1 - x0) * S, (ytop - ybot) * S
    X = lambda x: (x - x0) * S
    Yv = lambda y: (ytop - y) * S
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" font-family="Helvetica, Arial, sans-serif">',
           '<style>.lbl{font-size:10px;fill:#333;text-anchor:middle}.it{font-style:italic}.lg{font-size:11px;fill:#333}.ph{font-size:10px;fill:#777;text-anchor:middle;letter-spacing:1px;text-transform:uppercase}</style>']
    if SHOW_PHASES:
        y0, y1 = ybot_c - 0.12, ytop_c + 0.4
        for n, (name, bx0, bx1) in enumerate(band_boxes()):
            if n % 2 == 1:
                out.append(f'<rect x="{X(bx0):.1f}" y="{Yv(y1):.1f}" width="{(bx1-bx0)*S:.1f}" height="{(y1-y0)*S:.1f}" fill="#f5f5f5"/>')
            xm = X((bx0 + bx1) / 2); yh = Yv(y1 + 0.12)
            out.append(f'<text class="ph" x="{xm:.1f}" y="{yh:.1f}">{name}</text>')
    for i in draw_order:
        ch = edges[i][0]
        d = rounded_path([(X(x), Yv(y)) for x, y in paths[i]], [r * S for r in radii[i]])
        out.append(f'<path d="{d}" fill="none" stroke="#{COLOURS[ch]}" stroke-width="{LW*S:.1f}" stroke-linejoin="round"/>')
    for sid, (x, cy, w, h) in body.items():
        kind = marker_kind(sid)
        if kind == "terminus":
            out.append(f'<line x1="{X(x):.1f}" y1="{Yv(cy+h/2-0.05):.1f}" x2="{X(x):.1f}" y2="{Yv(cy-h/2+0.05):.1f}" stroke="#{STATION_STROKE}" stroke-width="{BAR_W*S:.1f}" stroke-linecap="round"/>')
        else:
            fill = f"#{STATION_STROKE}" if kind == "origin" else "#fff"
            out.append(f'<rect x="{X(x-w/2):.1f}" y="{Yv(cy+h/2):.1f}" width="{w*S:.1f}" height="{h*S:.1f}" rx="{ST_RX*S:.1f}" fill="{fill}" stroke="#{STATION_STROKE}" stroke-width="1"/>')
        lines, lh = [], 12
        for raw in STATIONS[sid][2]:          # wrap long labels for the wider sans font
            it = raw.startswith("*"); words = raw.lstrip("*").split(); cur = ""
            for wd in words:
                if len(cur) + len(wd) + 1 > 18 and cur:
                    lines.append(("*" if it else "") + cur); cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            lines.append(("*" if it else "") + cur)
        lvl = STATIONS[sid][1]
        if marker_kind(sid) == "terminus":
            top = Yv(cy) - (len(lines) - 1) * lh / 2 + 4
            for k, t in enumerate(lines):
                out.append(f'<text class="lbl" style="text-anchor:start" x="{X(x + BAR_W/2 + 0.12):.1f}" y="{top + k*lh:.1f}">{t.lstrip("*")}</text>')
            continue
        if lvl == 0:
            top = Yv(cy + h/2 + LABEL_GAP) - (len(lines) - 1) * lh - 3
        elif lvl == 2:
            top = Yv(cy - h/2 - LABEL_GAP) + 10
        else:
            top = Yv(cy - h/2 - LABEL_GAP) + 10
        for k, s in enumerate(lines):
            cls = "lbl it" if s.startswith("*") else "lbl"
            out.append(f'<text class="{cls}" x="{X(x):.1f}" y="{top + k*lh:.1f}">{s.lstrip("*")}</text>')
    if SHOW_LEGEND:
        ylg = ytop_c - (ytop_c - ybot) + (len(COLOURS) - 1) * LG_ROW + 0.55
        xl = X(band_boxes()[0][1])
        for k, ch in enumerate(sorted(COLOURS)):                           # preview only: title width is estimated
            yk = Yv(ylg - k * LG_ROW)
            tw = 5.6 * len(CH_TITLES[ch])
            out.append(f'<text class="lg" x="{xl:.1f}" y="{yk:.1f}" fill="#777">Ch. {ch}</text>')
            out.append(f'<text class="lg" x="{xl+0.75*S:.1f}" y="{yk:.1f}">{CH_TITLES[ch]}</text>')
            out.append(f'<line x1="{xl+0.75*S:.1f}" y1="{yk+4:.1f}" x2="{xl+0.75*S+tw:.1f}" y2="{yk+4:.1f}" stroke="#{COLOURS[ch]}" stroke-width="{LW*S:.1f}"/>')
    out.append("</svg>")
    return "\n".join(out)

# ================================================================ PNG (preview) backend
def write_png(path, dpi=200, use_tex=False):
    import subprocess, tempfile, os, shutil
    if use_tex:
        with tempfile.TemporaryDirectory() as d:
            tex = os.path.join(d, "m.tex")
            open(tex, "w").write(emit_tikz())
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", d, tex],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(os.path.join(d, "m.pdf")):
                sys.exit("pdflatex failed; see " + os.path.join(d, "m.log"))
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                            os.path.join(d, "m.pdf"), os.path.join(d, "m")], check=True)
            shutil.copy(os.path.join(d, "m.png"), path)
    else:
        try:
            import cairosvg
        except ImportError:
            sys.exit("PNG output needs cairosvg (pip install cairosvg) or use --tex with pdflatex/pdftoppm")
        cairosvg.svg2png(bytestring=emit_svg().encode(), write_to=path, scale=dpi / 96.0, background_color="white")
    print(f"wrote {path}", file=sys.stderr)

if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "tikz"
    dpi = int(args[args.index("--dpi") + 1]) if "--dpi" in args else 200
    use_tex = "--tex" in args
    pos = [a for a in args[1:] if not a.startswith("--") and a != str(dpi)]
    if mode == "svg":
        print(emit_svg())
    elif mode == "png":
        write_png(pos[0] if pos else "thesis_map.png", dpi, use_tex)
    elif mode == "all":
        base = pos[0] if pos else "thesis_map"
        open(base + ".tex", "w").write(emit_tikz())
        open(base + ".svg", "w").write(emit_svg())
        write_png(base + ".png", dpi, use_tex)
        print(f"wrote {base}.tex {base}.svg", file=sys.stderr)
    else:
        print(emit_tikz())