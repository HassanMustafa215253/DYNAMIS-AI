"""
GPU-resident N-body simulation, v5 (v4 smooth-flow release + state-of-the-bodies diagnostics).

What changed vs v4
  * The CPU diagnostics from the original code are ported to the GPU class (GPUNBody), vectorised in torch:
        orbital_energy()            pairwise  0.5 v_rel^2 - G (mi + mj) / r         -> (matrix, ids)
        pairwise_energy()           pairwise  0.5 mu v_rel^2 - G mi mj / r           -> (matrix, ids)   (< 0 means bound)
        angular_momentum()          per body  m (x vy - y vx)                        -> (n,) tensor
        tidal_perturbation(p)       pairwise  (mj / m_p) (r_i / r_j)^3 about primary -> (matrix, ids)
        kinetic_energy_per_body()   per body  0.5 m v^2
        potential_energy_per_body() per body  softened, each pair split half/half   (blocked, O(N^2) time, O(block) memory)
        total_energy_per_body()     KE + PE, sums to the total energy
        pairwise_linked_groups()    components of the "pairwise energy < 0" graph. A SEED list only: pairwise
                                    bound does not imply the whole group is bound (A-B and B-C can both be
                                    pairwise bound while A, B, C together are not)
        bound_groups()              the real classification: seeds are peeled (most unbound member first) until
                                    every member has 1/2|v_i - V_com|^2 + phi_i < 0 against the GROUP and the group
                                    energy T_com + W < 0; stripped bodies are re-linked and re-tested. Returns
                                    BoundGroup objects (mass, COM, T, W, E, 2T/|W|, r_half, tidal ratio, ...)
        calculate_dynamics()        prints the table above (same name as the CPU version)
        stability_report(monitor)   is the system settled? every body in a bound structure that passes the
                                    Mardling-Aarseth (hierarchical) or mutual-Hill (planet-like) criterion, or
                                    on a confirmed escape trajectory, with nothing changing over a long window
                                    (StabilityMonitor.record(sim) supplies the history). Verdict: SETTLED /
                                    PENDING / NOT SETTLED, plus a per-body fate. Refuses to certify a run whose
                                    energy drifted (dE/E > StabilityParams.energy_tol).
  * Matrix-valued diagnostics only cover LIVE bodies (merged-away bodies are dropped instead of filled with NaN).
    The returned `ids` tensor maps row/column k of the matrix to the original body id.
    They allocate an n x n matrix, so they refuse to run above `GPUNBody.max_matrix_n` (default 6000).
    The per-body quantities and bound_groups() are blocked and work at any N (the seed stage and the potential sums are O(N^2); peeling a seed of k bodies is O(k^2)).
  * Keys: B print bound groups, D print energy / angular momentum / virial summary.
  * Optional logging: --group-every F writes simulation_groups.csv (time, n_groups, largest, bodies_in_groups; confirmed groups);
    simulation_energy.csv gains an angular_momentum_z column.
  * python gpu_main.py --check compares every new diagnostic with a NumPy port of the original CPU formulas.

Note on angular momentum: merges conserve mass and linear momentum but the bodies have no spin, so the merged
cluster's internal angular momentum is lost. Total L is conserved between merges, not across them.

Install the fast path (optional, check your CUDA major version first):
    python -c "import torch; print(torch.version.cuda)"
    pip install cupy-cuda12x          # if the line above printed 12.x   (cupy-cuda11x for 11.x)
    pip install "cupy-cuda12x[ctk]"   # use this form if CuPy complains about missing CUDA DLLs

Front end (this version)
    Buttons (top-left): Labels OFF/ID/NAME, Live print, Snapshot, Stability, Speed -/+, Pause.
    Top-right panel:    simulated time, "run until t = ..." (how long to keep the run going before the outcome is
                        settled), alive bodies. The estimate needs the stability monitor (on for N <= 1000).
    Terminal:           only merge events, the short reports you ask for, and a short final verdict.
                        --verbose restores the old detailed output.
    Headless (no window, much faster):  python nbody_headless.py -h     (or: python gpu_main.py --headless ...)
    Scene: edit build_scene() below (the addCelestial calls); the window and the headless runner both use it.

Usage
    python gpu_main.py                          3-body merge demo
    python gpu_main.py --random 5000
    python gpu_main.py --bench                  headless timing table, eager vs fused
    python gpu_main.py --check                  headless check of the diagnostics against the NumPy formulas
    python gpu_main.py --kernel eager           force the PyTorch path
    python gpu_main.py --fp64-force             forces in float64 (slow on consumer GPUs)
    python gpu_main.py --set-high-perf-gpu      (Windows) render python.exe on the NVIDIA GPU
Keys: SPACE pause, F fullscreen, E fixed-dt, P debug print,
      UP/DOWN simulation speed x1.5 / /1.5, LEFT/RIGHT frame step budget, A toggle HUD,
      T toggle trail fade (start faded with --trail-fade), B bound groups, D diagnostics,
      S long-term stability report (settled / pending / not settled; needs the monitor, see --stability-every)
"""
import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import arcade
import arcade.gui
import numpy as np
import torch

AU_p_Pix = 1 / 50      # 1 AU = 50 pixels
G = 39.478             # AU^3 / (M_sun * year^2)
SOFTENING = 1e-5
EPS2 = SOFTENING ** 2
global_center_x = 0
global_center_y = 0

TRAIL_COLORS = [arcade.color.RED, arcade.color.BLUE, arcade.color.GREEN, arcade.color.YELLOW,
                arcade.color.MAGENTA, arcade.color.PINK, arcade.color.PURPLE]

_init_pos, _init_vel, _init_mass, _init_radius, _init_color, _init_name = [], [], [], [], [], []
RANDOM_N = 0

# Terminal output policy: warnings, merge events and the final verdict are always printed; set-up chatter and the
# long reports only with --verbose (VERBOSE = True).
VERBOSE = False


def say(*args, **kw):
    """print() that only speaks in --verbose mode."""
    if VERBOSE:
        print(*args, **kw)

# ====================================================================================== CUDA kernels
FORCE_SRC = r"""
extern "C" __global__ void force_kernel(
    const REAL* __restrict__ px, const REAL* __restrict__ py, const REAL* __restrict__ m,
    const int n, const REAL eps2, const REAL G,
    REAL* __restrict__ ax, REAL* __restrict__ ay, REAL* __restrict__ phi,
    REAL* __restrict__ minr2, int* __restrict__ minj)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const REAL mi = m[i];
    if (!(mi > (REAL)0)) {
        ax[i] = (REAL)0; ay[i] = (REAL)0; phi[i] = (REAL)0; minr2[i] = REAL_INF; minj[i] = 0;
        return;
    }
    const REAL xi = px[i], yi = py[i];
    REAL sx = (REAL)0, sy = (REAL)0, sp = (REAL)0, best = REAL_INF;
    int bj = 0;
    for (int j = 0; j < n; ++j) {
        const REAL mj = m[j];
        const REAL dx = px[j] - xi;
        const REAL dy = py[j] - yi;
        const REAL r2 = dx * dx + dy * dy;
        const REAL inv = RSQRT(r2 + eps2);
        const REAL w = mj * inv * inv * inv;
        sx += dx * w;
        sy += dy * w;
        if (j != i) {
            sp += mj * inv;
            if (mj > (REAL)0 && r2 < best) { best = r2; bj = j; }
        }
    }
    ax[i] = G * sx; ay[i] = G * sy; phi[i] = sp; minr2[i] = best; minj[i] = bj;
}
"""

TARGET_SRC = r"""
extern "C" __global__ void target_kernel(
    const REAL* __restrict__ px, const REAL* __restrict__ py, const REAL* __restrict__ rad_au,
    const REAL* __restrict__ m, const long long* __restrict__ rank, const int n,
    long long* __restrict__ best)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!(m[i] > (REAL)0)) { best[i] = -1; return; }
    long long b = rank[i];                       /* a live body always "hits" itself */
    const REAL xi = px[i], yi = py[i], ri = rad_au[i];
    for (int j = 0; j < n; ++j) {
        if (j != i && m[j] > (REAL)0) {
            const REAL dx = px[j] - xi, dy = py[j] - yi;
            const REAL s = ri + rad_au[j];
            if (dx * dx + dy * dy < s * s) {
                const long long rj = rank[j];
                if (rj > b) b = rj;
            }
        }
    }
    best[i] = b;
}
"""

PRELUDE_F32 = "#define REAL float\n#define RSQRT rsqrtf\n#define REAL_INF __int_as_float(0x7f800000)\n"
PRELUDE_F64 = ("#define REAL double\n#define RSQRT rsqrt\n"
               "#define REAL_INF __longlong_as_double(0x7ff0000000000000LL)\n")


def _prelude(dtype):
    return PRELUDE_F32 if dtype == torch.float32 else PRELUDE_F64


# ====================================================================================== adaptive dt (CPU)
# --- BEGIN compute_dt
def compute_dt(stats, dt_prev, p):
    """stats = [closest r^2, max|v|^2, max|a|^2, |dv|^2 of closest pair, |da|^2 of closest pair, r_max]."""
    r2, v2, a2, rv2, ra2, _ = stats
    inf = float("inf")
    min_dist = max(math.sqrt(r2), 1e-12) if math.isfinite(r2) else inf
    vmax, amax = math.sqrt(v2), math.sqrt(a2)
    dt_vel = 0.2 * min_dist / vmax if vmax > 0 else inf
    dt_acc = 0.2 * math.sqrt(min_dist / amax) if amax > 0 else inf
    cands = [p["dt_max"], dt_vel, dt_acc]
    if min_dist < p["close_threshold"]:
        rel_v, rel_a = math.sqrt(rv2), math.sqrt(ra2)
        c = []
        if rel_v > 0:
            c.append(p["close_safety"] * min_dist / rel_v)
        if rel_a > 0:
            c.append(p["close_safety"] * math.sqrt(min_dist / rel_a))
        cands.append(max(p["close_min_dt"], min(c)) if c else p["close_min_dt"])
    target = max(p["dt_min"], min(cands)) / p.get("dt_div", 3.0)   # dt_div = safety divisor (3 = original behaviour)
    return min(target, dt_prev * 1.2)
# --- END compute_dt


# ====================================================================================== eager (PyTorch) tile
def _force_tile(px_i, py_i, pen_i, ii, px, py, m, pen):
    """Fallback path: one block of rows i against all columns j (several un-fused kernels)."""
    dx = px[None, :] - px_i[:, None]
    dy = py[None, :] - py_i[:, None]
    r2 = dx * dx + dy * dy
    inv = (r2 + EPS2).rsqrt()
    rows = torch.arange(r2.shape[0], device=r2.device)
    inv[rows, ii] = 0.0                                   # no self term (potential must exclude it)
    phi = torch.einsum("bn,n->b", inv, m)
    w = inv * inv * inv * m[None, :]
    ax = G * torch.einsum("bn,bn->b", dx, w)
    ay = G * torch.einsum("bn,bn->b", dy, w)
    rr = r2 + pen_i[:, None] + pen[None, :]               # pen = +inf for dead bodies
    rr[rows, ii] = float("inf")
    mn, am = rr.min(dim=1)
    return ax, ay, phi, mn, am


@dataclass
class BoundGroup:
    """One confirmed gravitationally bound group. Units: M_sun, AU, yr (energies in M_sun AU^2 / yr^2).
    Potentials use the same softening as the integrator, so energies are consistent with GPUNBody.energy()."""
    ids: list              # original body ids, sorted
    mass: float
    com_pos: tuple         # centre of mass (x, y)
    com_vel: tuple         # centre-of-mass velocity (vx, vy)
    kinetic: float         # T, measured in the group's own COM frame
    potential: float       # W, mutual gravitational energy of the members only (no outside bodies)
    energy: float          # T + W  (< 0 for every reported group)
    virial_ratio: float    # 2T/|W|: 1 = virial equilibrium, < 2 = bound
    r_half: float          # half-mass radius about the COM
    r_max: float           # largest member distance from the COM
    tidal_ratio: float     # max over outside bodies k of (m_k / M)(r_max / D_k)^3   (nan if tidal=False)
    tidal_source: int      # id of the outside body giving that maximum (-1 if there is none)
    perturbed: bool        # tidal_ratio > tidal_threshold: the group reaches beyond some outsider's Hill sphere
    n_linked: int          # size of the pairwise-linked seed this group was peeled from
    n_stripped: int        # seed members removed by the unbinding (n_linked - size)

    @property
    def size(self):
        return len(self.ids)


# ====================================================================================== long-term stability
# What "settled" means here: every body is either (a) a member of a bound structure that passes an established
# stability criterion, or (b) on a confirmed escape trajectory, AND nothing has changed (no merger, no change of
# groups, no new escaper, no drifting orbit) for a long observation window. It is NOT a Lyapunov/chaos measure
# (a chaotic system can be permanently bound, a regular one can be unbound), and no finite computation can prove
# stability for unbounded time; the criteria are the ones used in the literature:
#   * hierarchical systems:  Mardling & Aarseth (2001) triple-stability criterion, applied at every level of the
#                            binary tree of the group (inner binary vs. the body/composite orbiting it)
#   * planet-like systems:   one body holds >= planetary_mass_fraction of the group mass -> mutual-Hill-radius
#                            spacing (Gladman 1993: 2*sqrt(3) for two planets; >~10 empirically for three or more,
#                            Chambers, Wetherill & Boss 1996)
#   * escape:                positive energy in the system COM frame, moving away from the core, beyond
#                            escape_factor x core radius, and the same on every snapshot of the quiet window
@dataclass
class StabilityParams:
    escape_factor: float = 3.0              # escapers must be farther than this x the core group's r_max
    min_dyn: float = 10.0                   # quiet window >= min_dyn x initial dynamical time ...
    min_periods: float = 50.0               # ... and >= min_periods x the longest bound orbital period
    a_tol: float = 0.10                     # allowed (max - min)/mean of a node's osculating a over the window
    margin_stable: float = 1.2              # criterion margin (actual / critical) needed to call a group stable
    planetary_mass_fraction: float = 0.9    # central-body mass fraction that selects the planetary test
    delta_two: float = 2.0 * math.sqrt(3.0)  # Hill-stability limit, two planets (mutual Hill radii)
    delta_multi: float = 10.0               # empirical spacing for three or more planets (mutual Hill radii)
    max_hier: int = 12                      # larger groups are clusters: no hierarchical analysis
    energy_tol: float = 1e-3                # max |dE/E| since the last merger before the run is deemed untrustworthy


@dataclass
class GroupStability:
    ids: list
    kind: str            # 'binary' | 'hierarchical' | 'planetary' | 'cluster'
    verdict: str         # 'stable' | 'marginal' | 'unstable' | 'unresolved'
    margin: float        # smallest (actual / critical) over the tests that apply (nan if none)
    reason: str
    p_max: float         # longest bound orbital period inside the group (nan if none)
    nodes: dict          # {tuple(member ids): (a, e)} osculating elements of every orbit, for the history tracker


def _kepler(mA, mB, r, v):
    """Osculating planar two-body elements of the relative orbit of B about A (r, v = relative position/velocity)."""
    mu = G * (mA + mB)
    rr = math.hypot(r[0], r[1])
    eps = 0.5 * (v[0] * v[0] + v[1] * v[1]) - mu / rr
    h = r[0] * v[1] - r[1] * v[0]
    e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h * h / (mu * mu)))
    bound = eps < 0.0
    a = -mu / (2.0 * eps) if bound else float("nan")
    return dict(a=a, e=e, bound=bound, sense=1 if h >= 0 else -1, mu=mu, r=rr,
                period=(2.0 * math.pi * math.sqrt(a ** 3 / mu)) if bound else float("inf"))


def _ma_margin(inner, m_in, outer, m_out):
    """Mardling-Aarseth (2001) margin of an inner binary (elements `inner`, total mass m_in) perturbed by a body of mass
    m_out on the orbit `outer` about the binary's COM: (outer pericentre / inner a) / critical ratio, with
        critical = 2.8 [(1 + q)(1 + e_out) / sqrt(1 - e_out)]^(2/5) (1 - 0.3 i / 180 deg),   q = m_out / m_in.
    The problem is planar, so i = 0 (same orbital sense) or 180 deg (opposite sense). > 1 means stable."""
    e = outer["e"]
    if not (inner["bound"] and outer["bound"]) or e >= 1.0:
        return 0.0
    q = m_out / m_in
    f_inc = 1.0 if inner["sense"] == outer["sense"] else 0.7
    crit = 2.8 * ((1.0 + q) * (1.0 + e) / math.sqrt(1.0 - e)) ** 0.4 * f_inc
    return outer["a"] * (1.0 - e) / inner["a"] / crit


class _TNode:
    __slots__ = ("ids", "m", "x", "v", "kids", "orb")

    def __init__(self, ids, m, x, v, kids=(), orb=None):
        self.ids, self.m, self.x, self.v, self.kids, self.orb = ids, m, x, v, kids, orb


def _build_tree(ids, m, x, v):
    """Binary tree of a group: repeatedly replace the two closest nodes by one composite node at their COM."""
    nodes = [_TNode((int(i),), float(mi), np.asarray(xi, float), np.asarray(vi, float))
             for i, mi, xi, vi in zip(ids, m, x, v)]
    while len(nodes) > 1:
        best = (math.inf, 0, 1)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                d = float(np.hypot(*(nodes[i].x - nodes[j].x)))
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        A, B = nodes[i], nodes[j]
        mt = A.m + B.m
        node = _TNode(tuple(sorted(A.ids + B.ids)), mt, (A.m * A.x + B.m * B.x) / mt,
                      (A.m * A.v + B.m * B.v) / mt, (A, B), _kepler(A.m, B.m, B.x - A.x, B.v - A.v))
        nodes = [n for k, n in enumerate(nodes) if k not in (i, j)] + [node]
    return nodes[0]


def _verdict_from_margin(margin, params):
    if margin >= params.margin_stable:
        return "stable"
    return "marginal" if margin >= 1.0 else "unstable"


def _assess_group(ids, m, x, v, params, perturbed=False, source=-1):
    """Stability of one bound group from its current state (arrays of the members only)."""
    ids = [int(i) for i in ids]
    k = len(ids)
    nodes, periods, margin = {}, [], float("nan")
    if k > params.max_hier:
        return GroupStability(ids, "cluster", "unresolved", margin,
                              f"{k} members: too many for a hierarchical analysis (a cluster keeps relaxing and "
                              "losing members slowly)", float("nan"), nodes)
    if k == 2:
        o = _kepler(m[0], m[1], x[1] - x[0], v[1] - v[0])
        nodes[tuple(sorted(ids))] = (o["a"], o["e"])
        periods.append(o["period"])
        kind = "binary"
        verdict = "stable" if o["bound"] else "unstable"
        reason = (f"isolated Kepler orbit, a = {o['a']:.4g} AU, e = {o['e']:.3f}, P = {o['period']:.4g} yr"
                  if o["bound"] else "the two bodies are not on a bound orbit")
    elif m.max() / m.sum() >= params.planetary_mass_fraction:
        kind = "planetary"
        c = int(np.argmax(m))
        sat = [i for i in range(k) if i != c]
        orbs, unbound = [], []
        for i in sat:
            o = _kepler(m[c], m[i], x[i] - x[c], v[i] - v[c])
            if not o["bound"]:
                unbound.append(ids[i])
            else:
                nodes[tuple(sorted((ids[c], ids[i])))] = (o["a"], o["e"])
                periods.append(o["period"])
                orbs.append((o["a"], i, o))
        if unbound:
            verdict, margin = "unstable", 0.0
            reason = f"body {unbound[0]} is not on a bound orbit about the central body {ids[c]}"
        else:
            orbs.sort(key=lambda t: t[0])
            crit = params.delta_two if len(sat) == 2 else params.delta_multi
            worst = None
            for (a1, i1, o1), (a2, i2, o2) in zip(orbs, orbs[1:]):
                rh = ((m[i1] + m[i2]) / (3.0 * m[c])) ** (1.0 / 3.0) * 0.5 * (a1 + a2)
                delta = (a2 * (1.0 - o2["e"]) - a1 * (1.0 + o1["e"])) / rh
                if worst is None or delta < worst[0]:
                    worst = (delta, ids[i1], ids[i2])
            margin = worst[0] / crit
            verdict = _verdict_from_margin(margin, params)
            reason = (f"planetary: central body {ids[c]}, closest pair {worst[1]}-{worst[2]} separated by "
                      f"{worst[0]:.2f} mutual Hill radii (needs >= {crit:.3g}; margin {margin:.3g})")
    else:
        kind = "hierarchical"
        root = _build_tree(ids, m, x, v)
        tests, unbound, stack = [], [], [root]
        while stack:
            n = stack.pop()
            if not n.kids:
                continue
            A, B = n.kids
            nodes[n.ids] = (n.orb["a"], n.orb["e"])
            periods.append(n.orb["period"])
            if not n.orb["bound"]:
                unbound.append(n.ids)
            for C, D in ((A, B), (B, A)):
                if C.kids:
                    tests.append((_ma_margin(C.orb, C.m, n.orb, D.m), C.ids, D.ids))
            stack.extend((A, B))
        if unbound:
            verdict, margin = "unstable", 0.0
            reason = f"the orbit at level {unbound[0]} is not bound (the group is not a stable hierarchy)"
        else:
            margin, ci, di = min(tests)
            verdict = _verdict_from_margin(margin, params)
            reason = (f"hierarchical (Mardling-Aarseth): inner {ci} vs outer {di}, margin {margin:.3g} "
                      f"(> 1 stable; verdict needs >= {params.margin_stable:g})")
    if perturbed and verdict == "stable":
        verdict = "marginal"
        reason += f"; the group reaches past the Hill sphere of outside body {source}"
    finite = [p for p in periods if math.isfinite(p)]
    return GroupStability(ids, kind, verdict, margin, reason, max(finite) if finite else float("nan"), nodes)


def _ids_str(ids, n=8):
    ids = list(ids)
    return str(ids) if len(ids) <= n else str(ids[:n])[:-1] + f", ... +{len(ids) - n}]"


@dataclass
class StabilityReport:
    t: float
    n_live: int
    core: list                 # ids of the core object (the most massive group or body)
    groups: list               # GroupStability for every confirmed bound group
    escaping: dict             # {ids tuple: dict(D, e, v)} objects passing the instantaneous escape test
    unresolved: list           # ids tuples of objects that are neither core, nor escaping
    fates: dict                # body id -> status string
    p_max: float               # longest bound orbital period in the system (nan if none)
    t_dyn: float               # dynamical time of the system now
    verdict: str = ""          # 'SETTLED' | 'PENDING' | 'NOT SETTLED'
    reasons: list = field(default_factory=list)
    quiet_span: float = 0.0    # time since the last observed change of structure
    required_span: float = 0.0
    n_records: int = 0
    observed_span: float = 0.0
    energy: float = float("nan")      # total energy now (softened, as the integrator)
    r_rms: float = float("nan")       # rms distance of the bodies from the system COM
    energy_drift: float = float("nan")  # |dE/E| since the last merger (needs history)

    @property
    def settled(self):
        return self.verdict == "SETTLED"

    def counts(self):
        c = {}
        for s in self.fates.values():
            c[s] = c.get(s, 0) + 1
        return c

    def summary(self):
        L = [f"Long-term stability check  t = {self.t:.4f} yr  live bodies = {self.n_live}",
             f"  VERDICT: {self.verdict}"]
        L += [f"    - {r}" for r in self.reasons]
        L.append(f"  core: {_ids_str(self.core)}")
        for g in self.groups:
            mg = "" if g.margin != g.margin else f", margin {g.margin:.3g}"
            L.append(f"  group {_ids_str(g.ids)}: {g.kind}, {g.verdict}{mg}")
            L.append(f"      {g.reason}")
        if self.escaping:
            L.append("  escaping objects (leaving the core):")
            for ids, d in self.escaping.items():
                L.append(f"      {_ids_str(ids)}: distance {d['D']:.4g} AU, speed rel. core {d['v']:.4g} AU/yr, "
                         f"energy {d['e']:+.4g}")
        if self.unresolved:
            L.append("  unresolved (neither bound to the core structure nor on an escape trajectory): "
                     + ", ".join(_ids_str(u, 4) for u in self.unresolved[:8]))
        L.append("  bodies: " + ", ".join(f"{n} {s}" for s, n in sorted(self.counts().items())))
        period = f"{self.p_max:.4g} yr" if self.p_max == self.p_max else "n/a"
        if self.energy_drift == self.energy_drift:
            L.append(f"  energy drift since the last merger: {self.energy_drift:.2e} (relative)")
        L.append(f"  history: {self.n_records} snapshot(s) over {self.observed_span:.4g} yr; structure unchanged for "
                 f"{self.quiet_span:.4g} yr; required {self.required_span:.4g} yr "
                 f"(max of {self._min_dyn:g} dynamical times and {self._min_periods:g} x longest period {period})")
        L.append("  note: 'settled' = no further ejection or exchange expected; it is not a Lyapunov chaos measure, and "
                 "the criteria (Mardling & Aarseth 2001, Gladman 1993, Chambers et al. 1996) are empirical or "
                 "approximate, not a proof for unbounded time.")
        return "\n".join(L)

    def stop_estimate(self):
        """When can the run be stopped?  Returns (verdict, t_stop, remaining_yr).
        The structure must stay unchanged for `required_span`, so the earliest possible stop is
        t_stop = (time of the last structural change) + required_span.
        SETTLED -> t_stop is when it became settled (remaining 0); PENDING -> run until t_stop if nothing changes;
        NOT SETTLED -> t_stop is only a lower bound (something is still unstable / interacting)."""
        t_stop = (self.t - self.quiet_span) + self.required_span
        return self.verdict, t_stop, max(0.0, t_stop - self.t)

    def stop_line(self):
        """One-line answer to 'how long do I have to keep this running?'."""
        v, t_stop, rem = self.stop_estimate()
        if v == "SETTLED":
            return f"SETTLED: safe to stop now (settled since t = {t_stop:.3f} yr)"
        if v == "PENDING":
            return f"PENDING: run until t = {t_stop:.2f} yr (+{rem:.2f} yr) if nothing changes"
        return f"NOT SETTLED: cannot stop before t = {t_stop:.2f} yr, and only once the issues clear"

    def brief(self, namer=None, max_reasons=2):
        """Short terminal report (the full one is summary()). `namer` maps a body id to its display label."""
        nm = namer or str

        def names(ids, n=6):
            ids = list(ids)
            s = ", ".join(nm(i) for i in ids[:n])
            return "[" + s + (f", +{len(ids) - n} more" if len(ids) > n else "") + "]"

        counts = ", ".join(f"{n} {s}" for s, n in sorted(self.counts().items()))
        L = [f"t = {self.t:.3f} yr | alive = {self.n_live} ({counts})", f"  {self.stop_line()}"]
        for g in self.groups:
            L.append(f"  group {names(g.ids)}: {g.kind}, {g.verdict}")
        for ids, d in self.escaping.items():
            L.append(f"  escaping {names(ids)}: {d['D']:.3g} AU from the core, speed {d['v']:.3g} AU/yr")
        if self.unresolved:
            L.append("  still interacting: " + ", ".join(names(u, 4) for u in self.unresolved[:4]))
        if self.verdict != "SETTLED":
            for r in self.reasons[:max_reasons]:
                L.append("  - " + (r if len(r) <= 150 else r[:147] + "..."))
        if self.energy_drift == self.energy_drift:
            L.append(f"  energy drift since last merger: {self.energy_drift:.1e}")
        return "\n".join(L)

    _min_dyn: float = 10.0
    _min_periods: float = 50.0


@dataclass
class _Rec:
    t: float
    n_live: int
    energy: float
    sig: tuple
    escaping: dict       # ids tuple -> distance from the core
    nodes: dict          # node ids -> (a, e)
    margin: float


def _signature(n_live, groups, escaping, unresolved):
    return (int(n_live), frozenset(tuple(g.ids) for g in groups), frozenset(escaping), tuple(sorted(unresolved)))


class StabilityMonitor:
    """Collects compact snapshots of the structure (call record(sim) every so often; it is cheap for small N) so that
    stability_report() can demand a long, quiet observation window instead of judging a single instant."""

    def __init__(self, params=None, max_records=5000):
        self.params = params or StabilityParams()
        self.records = deque(maxlen=max_records)
        self.t_dyn0 = None
        self.r0 = None             # initial rms radius: length scale for the escape test

    def record(self, sim):
        rep = sim._stability_snapshot(self.params, self.r0)
        if self.t_dyn0 is None:
            self.t_dyn0, self.r0 = rep.t_dyn, rep.r_rms
        margins = [g.margin for g in rep.groups if g.margin == g.margin]
        nodes = {}
        for g in rep.groups:
            nodes.update(g.nodes)
        self.records.append(_Rec(rep.t, rep.n_live, rep.energy, _signature(rep.n_live, rep.groups, rep.escaping, rep.unresolved),
                                 {k: d["D"] for k, d in rep.escaping.items()}, nodes,
                                 min(margins) if margins else float("nan")))
        return rep


def _finalize_report(rep, params, monitor):
    """Verdict = structure of the current snapshot + what the monitor's history says about its persistence."""
    fail, pend = [], []
    if rep.n_live <= 1:
        rep.verdict, rep.reasons = "SETTLED", ["a single body: nothing to interact with"]
        return rep
    for g in rep.groups:
        if g.verdict != "stable":
            fail.append(f"group {_ids_str(g.ids, 5)} is {g.verdict}: {g.reason}")
    if rep.unresolved:
        fail.append(f"{len(rep.unresolved)} object(s) are neither bound to the core nor escaping "
                    f"(still interacting): {', '.join(_ids_str(u, 4) for u in rep.unresolved[:4])}")
    recs = list(monitor.records) if monitor is not None else []
    t_dyn0 = monitor.t_dyn0 if (monitor is not None and monitor.t_dyn0) else rep.t_dyn
    p_max = rep.p_max if rep.p_max == rep.p_max else 0.0
    rep.required_span = max(params.min_dyn * t_dyn0, params.min_periods * p_max)
    rep._min_dyn, rep._min_periods = params.min_dyn, params.min_periods
    cur = _signature(rep.n_live, rep.groups, rep.escaping, rep.unresolved)
    suffix = []
    for r in reversed(recs):
        if r.sig != cur:
            break
        suffix.append(r)
    suffix.reverse()
    rep.n_records = len(recs)
    rep.observed_span = (rep.t - recs[0].t) if recs else 0.0
    rep.quiet_span = (rep.t - suffix[0].t) if suffix else 0.0
    base = None
    for r in reversed(recs):
        if r.n_live != rep.n_live:
            break
        base = r
    if base is not None and base.energy != 0 and base.energy == base.energy:
        rep.energy_drift = abs((rep.energy - base.energy) / base.energy)
        if rep.energy_drift > params.energy_tol:
            fail.append(f"integration not trustworthy: total energy changed by {rep.energy_drift:.2g} (relative) since the "
                        f"last merger, limit {params.energy_tol:g}; use a smaller time step (e.g. lower dtp['dt_min']) "
                        "or raise StabilityParams.energy_tol")
    cur_nodes = {}
    for g in rep.groups:
        cur_nodes.update(g.nodes)
    for key, (a_now, _) in cur_nodes.items():
        a = [a_now] + [r.nodes[key][0] for r in suffix if key in r.nodes]
        spread = (max(a) - min(a)) / (sum(a) / len(a))
        if spread > params.a_tol:
            fail.append(f"orbit {_ids_str(key, 5)}: semi-major axis varied by {100 * spread:.1f}% over the quiet "
                        f"window (limit {100 * params.a_tol:.0f}%): the orbit is not settled")
    mins = [r.margin for r in suffix if r.margin == r.margin]
    if mins and min(mins) < 1.0:
        fail.append(f"a stability margin fell below 1 (to {min(mins):.3g}) during the quiet window")
    for key, d in rep.escaping.items():
        series = [r.escaping[key] for r in suffix if key in r.escaping] + [d["D"]]
        if any(b < a * (1.0 - 1e-3) for a, b in zip(series, series[1:])):
            fail.append(f"escaping object {_ids_str(key, 4)} is not receding monotonically")
    if rep.quiet_span < rep.required_span:
        why = "no history was supplied" if not recs else (
            f"structure unchanged for {rep.quiet_span:.4g} yr, {rep.required_span:.4g} yr required")
        pend.append(f"quiet window too short: {why}")
    if rep.escaping and not pend:
        pass                                                   # persistent for the whole window -> confirmed
    confirmed = not pend and not fail
    for ids, d in rep.escaping.items():
        for i in ids:
            rep.fates[i] = "escaped" if (rep.quiet_span >= rep.required_span and recs) else "escaping"
    rep.reasons = fail + pend
    rep.verdict = "NOT SETTLED" if fail else ("PENDING" if pend else "SETTLED")
    if confirmed and not rep.reasons:
        rep.reasons = ["every body is in a stable bound structure or has escaped, and nothing changed for the whole "
                       "required window"]
    return rep


class GPUNBody:
    max_matrix_n = 6000           # n x n diagnostics (orbital/pairwise/tidal) refuse to run above this many live bodies

    def __init__(self, pos, vel, mass, radius_px, device, dtype=torch.float64, force_dtype=None,
                 collision_scale=1.0, max_block_elems=8_000_000, use_compile=False, kernel="auto"):
        self.dev = device
        self.dtype = dtype
        self.force_dtype = force_dtype or dtype
        self.collision_scale = collision_scale
        self.max_block_elems = max_block_elems
        self.merge_safety = 1.5        # merge only runs if closest pair < safety * 2 * r_max (see below)

        def T(a):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=device).contiguous()

        self.pos, self.vel, self.mass, self.rad = T(pos), T(vel), T(mass), T(radius_px)
        n = self.pos.shape[0]
        self.ids = torch.arange(n, device=device)
        self.t = 0.0                                          # python floats: dt/t live on the CPU now

        self.dtp = dict(dt_min=1e-3, dt_max=20.0, close_threshold=0.75, close_safety=0.2, close_min_dt=1e-6,
                        dt_div=3.0)
        self.fixed_dt = False
        self.fixed_dt_value = 5e-5

        # merge events (t, [absorbed ids], [absorbing ids], [new masses]); drained by the front end.
        # Recording costs one small sync per merge call, so it is only on for small systems.
        self.merge_log = []
        self.track_merges = n <= 2000

        self._tile = _force_tile
        if use_compile:
            try:
                self._tile = torch.compile(_force_tile, dynamic=True)
                say("[compute] torch.compile enabled for the eager force tile")
            except Exception as e:  # pragma: no cover
                print("[compute] torch.compile unavailable:", e)

        self._fused = False
        if kernel != "eager" and device.type == "cuda":
            self._init_fused(strict=(kernel == "cupy"))

        self.acc, closest, self._phi = self._accel_and_closest()
        self._stats = self._closest_stats(closest)
        self.dt = compute_dt(self._stats, 0.01, self.dtp)

    # ------------------------------------------------------------------ fused kernel setup + self-check
    def _init_fused(self, strict=False):
        try:
            import cupy as cp
            self._cp = cp
            self._k_force = cp.RawKernel(_prelude(self.force_dtype) + FORCE_SRC, "force_kernel")
            self._k_target = cp.RawKernel(_prelude(self.dtype) + TARGET_SRC, "target_kernel")
            self._fused = True
            ok, msg = self._selfcheck()
            if not ok:
                raise RuntimeError("numerical self-check failed: " + msg)
            say("[compute] fused kernels compiled and verified against the PyTorch path:", msg)
        except Exception as e:
            self._fused = False
            print(f"[compute] WARNING: fused CUDA kernels not used (slower PyTorch path): {str(e).splitlines()[0]}"
                  "  -> pip install cupy-cuda12x  (match torch.version.cuda)")
            if strict:
                raise

    @property
    def backend_name(self):
        return "fused CUDA kernels" if self._fused else "PyTorch eager"

    def _selfcheck(self):
        g = torch.Generator(device="cpu").manual_seed(1)
        n = 700
        p = (torch.rand(n, 2, generator=g, dtype=torch.float64) * 8 - 4).to(self.dev, self.dtype)
        m = (torch.rand(n, generator=g, dtype=torch.float64) + 0.1).to(self.dev, self.dtype)
        m[torch.randperm(n, generator=g)[:90].to(self.dev)] = 0.0       # some dead bodies
        a0, c0, ph0 = self._accel_eager(p, m)
        a1, c1, ph1 = self._accel_fused(p, m)
        tol = 2e-3 if self.force_dtype == torch.float32 else 1e-9
        err_a = float((a0 - a1).abs().max() / a0.abs().max())
        err_p = float((ph0 * m - ph1 * m).abs().max() / (ph0 * m).abs().max())
        err_r = abs(float(c0[0]) - float(c1[0])) / float(c0[0])
        same_pair = sorted((int(c0[1]), int(c0[2]))) == sorted((int(c1[1]), int(c1[2])))
        # overlap-target kernel vs eager
        rad = (torch.rand(n, generator=g, dtype=torch.float64) * 0.3 + 0.1).to(self.dev, self.dtype)
        alive = m > 0
        order = torch.sort(m, stable=True).indices
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n, device=self.dev)
        b0 = self._best_rank_eager(p, rad, alive, rank)
        b1 = self._best_rank_fused(p, rad, m, rank)
        same_t = bool(torch.equal(b0, b1))
        msg = (f"acc err {err_a:.1e}, potential err {err_p:.1e}, closest-pair err {err_r:.1e}, "
               f"same pair {same_pair}, merge-target match {same_t}")
        return (err_a < tol and err_p < tol and err_r < tol and same_pair and same_t), msg

    # ------------------------------------------------------------------ helpers
    def _blocks(self, n):
        b = max(1, min(n, self.max_block_elems // max(n, 1)))
        return [(s, min(n, s + b)) for s in range(0, n, b)]

    def set_fixed_dt(self, flag):
        self.fixed_dt = flag
        if flag:
            self.dt = self.fixed_dt_value

    # ------------------------------------------------------------------ forces + potential + closest pair
    def _accel_and_closest(self):
        if self._fused:
            return self._accel_fused(self.pos, self.mass)
        return self._accel_eager(self.pos, self.mass)

    def _accel_eager(self, p, m):
        n = p.shape[0]
        fd = self.force_dtype
        alive = m > 0
        pen = torch.where(alive, torch.zeros_like(m), torch.full((), float("inf"), dtype=m.dtype,
                                                                  device=self.dev)).to(fd)
        px, py, mf = p[:, 0].to(fd).contiguous(), p[:, 1].to(fd).contiguous(), m.to(fd)
        ar = torch.arange(n, device=self.dev)
        axs, ays, phs, mns, ams = [], [], [], [], []
        for s, e in self._blocks(n):
            ax, ay, ph, mn, am = self._tile(px[s:e], py[s:e], pen[s:e], ar[s:e], px, py, mf, pen)
            axs.append(ax); ays.append(ay); phs.append(ph); mns.append(mn); ams.append(am)
        acc = torch.stack([torch.cat(axs), torch.cat(ays)], 1).to(self.dtype)
        acc = torch.where(alive[:, None], acc, torch.zeros_like(acc))
        mn = torch.cat(mns)
        bi = mn.argmin()
        return acc, (mn[bi].to(self.dtype), bi, torch.cat(ams)[bi]), torch.cat(phs)

    def _accel_fused(self, p, m):
        cp = self._cp
        n = p.shape[0]
        fd = self.force_dtype
        real = np.float32 if fd == torch.float32 else np.float64
        px, py, mf = p[:, 0].to(fd).contiguous(), p[:, 1].to(fd).contiguous(), m.to(fd).contiguous()
        ax, ay, ph, mn = (torch.empty(n, dtype=fd, device=self.dev) for _ in range(4))
        mj = torch.empty(n, dtype=torch.int32, device=self.dev)
        block = 128
        with cp.cuda.ExternalStream(torch.cuda.current_stream(self.dev).cuda_stream):
            self._k_force(((n + block - 1) // block,), (block,),
                          (cp.asarray(px), cp.asarray(py), cp.asarray(mf), np.int32(n), real(EPS2), real(G),
                           cp.asarray(ax), cp.asarray(ay), cp.asarray(ph), cp.asarray(mn), cp.asarray(mj)))
        acc = torch.stack([ax, ay], 1).to(self.dtype)         # dead rows already 0 from the kernel
        bi = mn.argmin()
        return acc, (mn[bi].to(self.dtype), bi, mj[bi].long()), ph

    # ------------------------------------------------------------------ per-step stats + dt (ONE small D2H)
    def _closest_stats(self, closest):
        best_r2, bi, bj = closest
        vel, acc = self.vel, self.acc                         # dead rows have v = a = 0
        dv, da = vel[bi] - vel[bj], acc[bi] - acc[bj]
        s = torch.stack([best_r2, (vel * vel).sum(1).max(), (acc * acc).sum(1).max(),
                         (dv * dv).sum(), (da * da).sum(), self.rad.max()])
        return s.cpu().tolist()

    # ------------------------------------------------------------------ merging
    def _best_rank_eager(self, p, r_au, alive, rank):
        n = p.shape[0]
        idx = torch.arange(n, device=self.dev)
        neg1 = torch.tensor(-1, dtype=torch.long, device=self.dev)
        best = torch.empty(n, dtype=torch.long, device=self.dev)
        for s, e in self._blocks(n):
            d = p[s:e, None, :] - p[None, :, :]
            d2 = (d * d).sum(-1)
            hit = d2 < (r_au[s:e, None] + r_au[None, :]) ** 2
            hit |= idx[s:e, None] == idx[None, :]
            hit &= alive[s:e, None] & alive[None, :]
            best[s:e] = torch.where(hit, rank[None, :], neg1).amax(dim=1)
        return best

    def _best_rank_fused(self, p, r_au, m, rank):
        cp = self._cp
        n = p.shape[0]
        px, py = p[:, 0].contiguous(), p[:, 1].contiguous()
        best = torch.empty(n, dtype=torch.long, device=self.dev)
        block = 128
        with cp.cuda.ExternalStream(torch.cuda.current_stream(self.dev).cuda_stream):
            self._k_target(((n + block - 1) // block,), (block,),
                           (cp.asarray(px), cp.asarray(py), cp.asarray(r_au.contiguous()),
                            cp.asarray(m.contiguous()), cp.asarray(rank), np.int32(n), cp.asarray(best)))
        return best

    def _merge(self):
        """Branch-free merge of every overlapping cluster (inelastic: mass + momentum conserved)."""
        p, v, m, rad = self.pos, self.vel, self.mass, self.rad
        n = p.shape[0]
        alive = m > 0
        idx = torch.arange(n, device=self.dev)
        r_au = rad * (AU_p_Pix * self.collision_scale)

        order = torch.sort(m, stable=True).indices
        rank = torch.empty_like(order)
        rank[order] = idx
        best = (self._best_rank_fused(p, r_au, m, rank) if self._fused
                else self._best_rank_eager(p, r_au, alive, rank))

        target = torch.where(alive, order[best.clamp(min=0)], idx)
        for _ in range(3):
            target = target[target]
        target = torch.where(target[target] == target, target, idx)

        fl = alive.to(m.dtype)
        cnt = torch.zeros_like(m).index_add_(0, target, fl)
        in_cl = alive & (cnt[target] > 1)
        is_root = target == idx
        win, lose = in_cl & is_root, in_cl & ~is_root

        zero = torch.zeros_like(m)
        w = torch.where(in_cl, m, zero)
        M = torch.zeros_like(m).index_add_(0, target, w)
        Pm = torch.zeros_like(p).index_add_(0, target, w[:, None] * p)
        Vm = torch.zeros_like(v).index_add_(0, target, w[:, None] * v)
        R3 = torch.zeros_like(m).index_add_(0, target, torch.where(in_cl, rad ** 3, zero))
        Msafe = torch.where(M > 0, M, torch.ones_like(M))

        self.pos = torch.where(win[:, None], Pm / Msafe[:, None],
                               torch.where(lose[:, None], torch.zeros_like(p), p))
        self.vel = torch.where(win[:, None], Vm / Msafe[:, None],
                               torch.where(lose[:, None], torch.zeros_like(v), v))
        self.mass = torch.where(win, M, torch.where(lose, zero, m))
        self.rad = torch.where(win, R3.pow(1.0 / 3.0), rad)

        if self.track_merges:
            li = lose.nonzero().squeeze(1)
            if li.numel():
                wi = target[li]
                self.merge_log.append((self.t + self.dt, self.ids[li].tolist(), self.ids[wi].tolist(),
                                       self.mass[wi].tolist()))

    def _maybe_merge(self):
        """
        dt is capped so that no pair can close by more than ~13% of the closest separation in one step,
        so a collision is only possible when the closest pair is within ~1.5x the contact distance.
        The stats come from the previous step (already on the CPU), so this check costs nothing.
        """
        if self.fixed_dt:                                      # no such guarantee with a fixed dt
            self._merge()
            return
        r2, rmax = self._stats[0], self._stats[5]
        reach = self.merge_safety * 2.0 * rmax * AU_p_Pix * self.collision_scale
        if r2 < reach * reach:
            self._merge()

    # ------------------------------------------------------------------ one sub-step
    def step(self):
        dt = self.dt
        self.vel += (0.5 * dt) * self.acc                      # kick
        self.pos += dt * self.vel                              # drift
        self._maybe_merge()                                    # merge BEFORE the force: second kick is consistent
        self.acc, closest, self._phi = self._accel_and_closest()
        self.vel += (0.5 * dt) * self.acc                      # kick
        self.t += dt
        self._stats = self._closest_stats(closest)             # the single sync of the sub-step
        if not self.fixed_dt:
            self.dt = compute_dt(self._stats, dt, self.dtp)

    # ------------------------------------------------------------------ diagnostics / transfers
    def energy(self):
        """Kinetic + softened potential. The potential is a by-product of the last force pass."""
        kin = 0.5 * (self.mass * (self.vel * self.vel).sum(1)).sum()
        pot = -0.5 * G * (self.mass * self._phi.to(self.dtype)).sum()
        return kin + pot

    def energy_reference(self):
        """Blocked eager N x N energy (slow); used only to validate the free energy."""
        p, v, m = self.pos, self.vel, self.mass
        n = p.shape[0]
        idx = torch.arange(n, device=self.dev)
        kin = 0.5 * (m * (v * v).sum(1)).sum()
        pot = torch.zeros((), dtype=self.dtype, device=self.dev)
        for s, e in self._blocks(n):
            d = p[s:e, None, :] - p[None, :, :]
            r = torch.sqrt((d * d).sum(-1) + EPS2)
            u = torch.where(idx[s:e, None] != idx[None, :], m[s:e, None] * m[None, :] / r,
                            torch.zeros((), dtype=self.dtype, device=self.dev))
            pot = pot + u.sum()
        return kin - 0.5 * G * pot

    def fetch_render(self, cx, cy):
        """One float32 transfer: (n, 4) = x_px, y_px, r_px (0 if dead), id."""
        alive = self.mass > 0
        x = self.pos[:, 0] / AU_p_Pix + cx
        y = self.pos[:, 1] / AU_p_Pix + cy
        r = torch.where(alive, self.rad, torch.zeros_like(self.rad))
        buf = torch.stack([x, y, r, self.ids.to(self.dtype)], 1).to(torch.float32).cpu().numpy()
        return buf, self.t, self.dt, int((buf[:, 2] > 0).sum())

    def fetch_log(self, k):
        """The k heaviest live bodies (or all live bodies if k is None), float64: x y vx vy m id."""
        n = self.pos.shape[0]
        if k is None or k >= n:
            idx = (self.mass > 0).nonzero().squeeze(1)
        else:
            idx = torch.topk(self.mass, k).indices
        rows = torch.cat([self.pos[idx], self.vel[idx], self.mass[idx, None],
                          self.ids[idx].to(self.dtype)[:, None]], dim=1)
        arr = rows.cpu().numpy()
        return arr[arr[:, 4] > 0]

    def compact(self):
        idx = (self.mass > 0).nonzero().squeeze(1)
        self.pos, self.vel = self.pos.index_select(0, idx), self.vel.index_select(0, idx)
        self.mass, self.rad = self.mass.index_select(0, idx), self.rad.index_select(0, idx)
        self.acc, self.ids = self.acc.index_select(0, idx), self.ids.index_select(0, idx)
        self._phi = self._phi.index_select(0, idx)

    # ====================================================================== body-state analysis (GPU ports)
    # Everything below works on the current state (self.pos / vel / mass). Dead bodies have m = 0 and are
    # excluded. Matrix results are over LIVE bodies only: row/column k <-> original body id ids[k].
    def _live(self):
        return (self.mass > 0).nonzero().squeeze(1)

    def _ablocks(self, n):
        """Row blocks for the analysis passes (smaller than _blocks: these build several temporaries)."""
        b = max(1, min(n, (self.max_block_elems // 4) // max(n, 1)))
        return [(s, min(n, s + b)) for s in range(0, n, b)]

    def _check_matrix(self, n, what):
        if n > self.max_matrix_n:
            raise ValueError(f"{what}: {n} live bodies would need an {n}x{n} matrix; raise "
                             f"GPUNBody.max_matrix_n (now {self.max_matrix_n}) or use the per-body / "
                             "bound_groups() diagnostics, which are blocked.")

    @staticmethod
    def _pdist(a):
        """Pairwise Euclidean distance matrix (direct differences, no matmul trick: exact for close pairs)."""
        return torch.cdist(a, a, compute_mode="donot_use_mm_for_euclid_dist")

    def orbital_energy(self):
        """E[i, j] = 0.5 |v_i - v_j|^2 - G (m_i + m_j) / r_ij, NaN on the diagonal. Returns (matrix, ids)."""
        idx = self._live()
        self._check_matrix(idx.numel(), "orbital_energy")
        m = self.mass[idx]
        r = self._pdist(self.pos[idx])
        v = self._pdist(self.vel[idx])
        with torch.no_grad():
            E = 0.5 * v * v - G * (m[:, None] + m[None, :]) / r
        E.fill_diagonal_(float("nan"))
        return E, self.ids[idx]

    def angular_momentum(self):
        """Per-body z angular momentum about the origin, m (x vy - y vx), shape (n,). Dead bodies give 0."""
        p, v = self.pos, self.vel
        return (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]) * self.mass

    def pairwise_energy(self):
        """Two-body energy in the pair's COM frame: 0.5 mu |dv|^2 - G m_i m_j / r. < 0 means the pair is
        bound. NaN on the diagonal. Returns (matrix, ids)."""
        idx = self._live()
        self._check_matrix(idx.numel(), "pairwise_energy")
        m = self.mass[idx]
        r = self._pdist(self.pos[idx])
        v = self._pdist(self.vel[idx])
        mul = m[:, None] * m[None, :]
        E = 0.5 * (mul / (m[:, None] + m[None, :])) * v * v - G * mul / r
        E.fill_diagonal_(float("nan"))
        return E, self.ids[idx]

    def tidal_perturbation(self, primary):
        """n[i, j] = (m_j / m_primary) (r_i / r_j)^3 with r_k the distance of body k from the primary.
        `primary` is a body id or any object with an .index attribute (e.g. a Celestial).
        Row/column of the primary and the diagonal are NaN. Returns (matrix, ids)."""
        pid = int(getattr(primary, "index", primary))
        idx = self._live()
        self._check_matrix(idx.numel(), "tidal_perturbation")
        ids = self.ids[idx]
        hit = (ids == pid).nonzero()
        if hit.numel() == 0:
            raise ValueError(f"body {pid} is not alive (merged away or unknown id)")
        k = int(hit[0, 0])
        p, m = self.pos[idx], self.mass[idx]
        r = torch.linalg.norm(p - p[k], dim=1)
        with torch.no_grad():
            n = (m[None, :] / m[k]) * (r[:, None] / r[None, :]) ** 3
        n[k, :] = float("nan")
        n[:, k] = float("nan")
        n.fill_diagonal_(float("nan"))
        return n, ids

    def kinetic_energy_per_body(self):
        return 0.5 * self.mass * (self.vel * self.vel).sum(1)          # dead bodies have m = 0

    def _member_potential(self, p, m):
        """phi_i = -G sum_{j != i} m_j / sqrt(r_ij^2 + eps^2) over the member set (p, m) only. Blocked, shape (k,).
        Softened like the force, so it is consistent with energy()."""
        k = p.shape[0]
        out = torch.zeros(k, dtype=self.dtype, device=self.dev)
        for s, e in self._ablocks(k):
            d = p[s:e, None, :] - p[None, :, :]
            r = torch.sqrt((d * d).sum(-1) + EPS2)
            U = m[None, :] / r
            rows = torch.arange(e - s, device=self.dev)
            U[rows, rows + s] = 0.0                                    # no self term
            out[s:e] = -G * U.sum(1)
        return out

    def potential_energy_per_body(self):
        """Softened potential (same softening as the force); each pair's energy is split half/half between
        its two bodies. Blocked, so memory stays bounded at any N. Shape (n,), dead bodies give 0."""
        V = torch.zeros_like(self.mass)
        idx = self._live()
        if idx.numel() < 2:
            return V
        m = self.mass[idx]
        V[idx] = 0.5 * m * self._member_potential(self.pos[idx], m)
        return V

    def total_energy_per_body(self):
        """Summed over all bodies this is the total energy of the system."""
        return self.kinetic_energy_per_body() + self.potential_energy_per_body()

    # ====================================================================== bound groups
    # Two bodies with negative two-body energy are NOT necessarily part of a bound group: in an N-body system the
    # other members change the energy budget (A-B and B-C can both be pairwise bound while A, B, C together are
    # not). So the analysis is done in two stages:
    #   1. SEED     connected components of the "pairwise two-body energy < 0" graph (pairwise_linked_groups).
    #               Cheap and only a candidate list: it says "linked by pairwise binding", not "bound group".
    #   2. UNBIND   inside each seed, test the members against the GROUP as a whole, in the group's own COM frame:
    #                   e_i = 1/2 |v_i - V_com|^2 + phi_i      phi_i = potential of body i from the other members
    #                   E   = T_com + W                        (T_com in the COM frame, W = mutual potential)
    #               A group is accepted only when every e_i < 0 (each member is bound to the rest) AND E < 0
    #               (the group as a whole is bound). Otherwise the member with the largest e_i is removed, the sums
    #               are updated incrementally, and the test repeats ("peeling", O(k^2) in total per seed).
    #               The removed bodies are linked again among themselves and processed the same way, so a bound
    #               sub-group (e.g. a binary that was only attached to the seed by a chain) is not lost.
    # Result: disjoint groups, each bound by its own self-gravity at this instant.
    #   3. MERGE    every group tries to absorb lone bodies and other groups (exact union energy, then the same
    #               collective test), so membership does not depend on orbital phase: the outer body of a hierarchical
    #               triple is attached even when, at this instant, it is pairwise bound to neither inner body.
    # Limits (inherent to any instantaneous energy criterion): the outside field is not part of E (it is reported as
    # a tidal ratio instead); sub-structure inside a group (a binary in a cluster) is not reported separately; and
    # an E < 0 snapshot is a necessary condition for lasting boundness, not a proof of long-term stability (see
    # stability_report for that).
    def _bound_labels(self, idx):
        """SEED graph. Connected components of {(i, j): pairwise energy < 0} over the bodies `idx`, computed
        on the GPU by blocked min-label propagation with pointer jumping (a few O(N^2) passes).
        Returns a label per body (the smallest member index of its component)."""
        p, v, m = self.pos[idx], self.vel[idx], self.mass[idx]
        k = idx.numel()
        lab = torch.arange(k, device=self.dev)
        big = torch.tensor(k, dtype=torch.long, device=self.dev)
        for _ in range(200):
            cand = torch.empty_like(lab)
            for s, e in self._ablocks(k):
                dx = p[s:e, None, :] - p[None, :, :]
                r = torch.sqrt((dx * dx).sum(-1) + EPS2)               # softened, as in the force
                dv = v[s:e, None, :] - v[None, :, :]
                v2 = (dv * dv).sum(-1)
                mm = m[s:e, None] * m[None, :]
                E = 0.5 * (mm / (m[s:e, None] + m[None, :])) * v2 - G * mm / r
                bound = E < 0
                rows = torch.arange(e - s, device=self.dev)
                bound[rows, rows + s] = False
                cand[s:e] = torch.where(bound, lab[None, :], big).amin(dim=1)
            mn = torch.minimum(lab, cand)
            root = lab.clone().scatter_reduce_(0, lab, mn, reduce="amin", include_self=True)
            new = torch.minimum(mn, root[lab])
            new = new[new]                                             # pointer jumping
            if torch.equal(new, lab):
                break
            lab = new
        return lab

    def _components(self, sel):
        """Connected components of the pairwise-bound graph restricted to the bodies `sel` (LongTensor of indices
        into the state arrays). Returns a list of sorted numpy index arrays, lone bodies included."""
        k = sel.numel()
        if k == 0:
            return []
        s = sel.cpu().numpy()
        lab = self._bound_labels(sel).cpu().numpy() if k > 1 else np.zeros(1, dtype=np.int64)
        order = np.argsort(lab, kind="stable")
        cuts = np.flatnonzero(np.diff(lab[order])) + 1
        return [np.sort(s[g]) for g in np.split(order, cuts)]

    def pairwise_linked_groups(self, min_size=1):
        """Connected components of the graph "pairwise two-body energy < 0" (what the CPU calculate_dynamics built).
        This is only a candidate list (the SEED stage), not a set of bound groups: see bound_groups().
        Returns lists of original body ids, largest first."""
        ids = self.ids.cpu().numpy()
        groups = [sorted(int(ids[i]) for i in c) for c in self._components(self._live())]
        groups.sort(key=lambda g: (-len(g), g[0]))
        return [g for g in groups if len(g) >= min_size]

    def _unbind(self, sel):
        """Peel the member set `sel` (LongTensor of state indices) until every member is bound to the rest and the
        group energy is negative. Returns a numpy bool mask over `sel` (True = still a member). If fewer than two
        members survive there is no group (the mask then has a single True)."""
        p, v, m = self.pos[sel], self.vel[sel], self.mass[sel]
        k = sel.numel()
        phi = self._member_potential(p, m)                              # negative; members only
        alive = torch.ones(k, dtype=torch.bool, device=self.dev)
        M = m.sum()
        P = (m[:, None] * v).sum(0)
        T_lab = 0.5 * (m * (v * v).sum(1)).sum()
        W = 0.5 * (m * phi).sum()
        n_alive = k
        ninf = torch.tensor(float("-inf"), dtype=self.dtype, device=self.dev)
        while n_alive >= 2:
            Vc = P / M
            dv = v - Vc
            e = torch.where(alive, 0.5 * (dv * dv).sum(1) + phi, ninf)
            emax, j = e.max(0)
            E = T_lab - 0.5 * M * (Vc * Vc).sum() + W                   # T_com + W
            emax_c, E_c, j_c = torch.stack([emax, E, j.to(self.dtype)]).tolist()   # one sync per iteration
            if emax_c < 0.0 and E_c < 0.0:
                break
            j = int(j_c)                                                # most unbound member goes first
            alive[j] = False
            n_alive -= 1
            W = W - m[j] * phi[j]                                       # drop j's pair terms
            phi = phi + G * m[j] / torch.sqrt(((p - p[j]) ** 2).sum(1) + EPS2)
            M = M - m[j]
            P = P - m[j] * v[j]
            T_lab = T_lab - 0.5 * m[j] * (v[j] * v[j]).sum()
        return alive.cpu().numpy()

    def _describe(self, s, n_linked, live, ids_np, tidal, tidal_threshold):
        """Fresh (non-incremental) statistics of the member set `s` (numpy state indices) as a BoundGroup."""
        sel = torch.as_tensor(s, device=self.dev)
        p, v, m = self.pos[sel], self.vel[sel], self.mass[sel]
        M = m.sum()
        X = (m[:, None] * p).sum(0) / M
        Vc = (m[:, None] * v).sum(0) / M
        dv = v - Vc
        T = 0.5 * (m * (dv * dv).sum(1)).sum()
        W = 0.5 * (m * self._member_potential(p, m)).sum()
        rs, order = torch.sort(torch.sqrt(((p - X) ** 2).sum(1)))
        cum = torch.cumsum(m[order], 0)
        r_half = rs[int((cum >= 0.5 * M).nonzero()[0, 0])]
        r_max = rs[-1]
        t_ratio, t_src = float("nan"), -1
        if tidal:
            ext = live[~torch.isin(live, sel)]
            if ext.numel():
                D = torch.sqrt(((self.pos[ext] - X) ** 2).sum(1))
                ratio = (self.mass[ext] / M) * (r_max / D) ** 3        # Hill criterion: group inside the Hill
                val, j = ratio.max(0)                                  # sphere of body k  <=>  ratio < 1/3
                t_ratio, t_src = float(val), int(ids_np[int(ext[int(j)])])
            else:
                t_ratio = 0.0
        T, W = float(T), float(W)
        return BoundGroup(
            ids=sorted(int(ids_np[i]) for i in s), mass=float(M),
            com_pos=(float(X[0]), float(X[1])), com_vel=(float(Vc[0]), float(Vc[1])),
            kinetic=T, potential=W, energy=T + W, virial_ratio=(2 * T / abs(W)) if W != 0 else float("nan"),
            r_half=float(r_half), r_max=float(r_max), tidal_ratio=t_ratio, tidal_source=t_src,
            perturbed=bool(t_ratio > tidal_threshold), n_linked=int(n_linked), n_stripped=int(n_linked - len(s)))

    def _merge_objects(self, found, max_try=8):
        """Phase-independent completion of the groups. found: list of (members, n_linked), members = numpy state
        indices. The pairwise seed stage can miss a body that is bound to a group as a whole but, at this orbital
        phase, to none of its members pairwise (e.g. the outer body of a hierarchical triple while the inner bodies
        move against it). So every group tries to absorb each lone body and each other group: the exact energy of the
        union is E_A + E_B + 1/2 mu |V_A - V_B|^2 + W_AB; the best candidates (E < 0) are verified with the full
        collective test (_unbind must keep every member). Repeats until nothing merges."""
        found = [(np.asarray(k), n) for k, n in found]
        if not found:
            return found
        live = self._live().cpu().numpy()
        n_state = self.pos.shape[0]

        def stats(idx):
            sel = torch.as_tensor(idx, device=self.dev)
            p, v, m = self.pos[sel], self.vel[sel], self.mass[sel]
            M = m.sum()
            Vc = (m[:, None] * v).sum(0) / M
            T = 0.5 * (m * ((v - Vc) ** 2).sum(1)).sum()
            W = 0.5 * (m * self._member_potential(p, m)).sum()
            return p, m, M, Vc, T + W

        while True:
            member = np.zeros(n_state, bool)
            for k, _ in found:
                member[k] = True
            singles = live[~member[live]]
            order = sorted(range(len(found)), key=lambda i: -float(self.mass[torch.as_tensor(found[i][0], device=self.dev)].sum()))
            merged = False
            for gi in order:
                kept, _ = found[gi]
                p, m, M, Vc, EG = stats(kept)
                cands = []
                if singles.size:
                    sg = torch.as_tensor(singles, device=self.dev)
                    pj, vj, mj = self.pos[sg], self.vel[sg], self.mass[sg]
                    phi = torch.empty(len(singles), dtype=self.dtype, device=self.dev)
                    step = max(1, (self.max_block_elems // 4) // max(len(kept), 1))
                    for a in range(0, len(singles), step):
                        d = pj[a:a + step, None, :] - p[None, :, :]
                        phi[a:a + step] = -G * (m[None, :] / torch.sqrt((d * d).sum(-1) + EPS2)).sum(1)
                    mu = mj * M / (mj + M)
                    Eu = EG + 0.5 * mu * ((vj - Vc) ** 2).sum(1) + mj * phi
                    for i in (Eu < 0).nonzero().squeeze(1).tolist():
                        cands.append((float(Eu[i]), "s", int(singles[i])))
                for gj in range(len(found)):
                    if gj == gi:
                        continue
                    p2, m2, M2, V2, E2 = stats(found[gj][0])
                    d = p[:, None, :] - p2[None, :, :]
                    Wc = -G * (m[:, None] * m2[None, :] / torch.sqrt((d * d).sum(-1) + EPS2)).sum()
                    Eu = EG + E2 + 0.5 * (M * M2 / (M + M2)) * ((Vc - V2) ** 2).sum() + Wc
                    if float(Eu) < 0:
                        cands.append((float(Eu), "g", gj))
                cands.sort(key=lambda c: c[0])
                for _, kind, j in cands[:max_try]:
                    union = np.sort(np.append(kept, j) if kind == "s" else np.concatenate([kept, found[j][0]]))
                    if self._unbind(torch.as_tensor(union, device=self.dev)).all():
                        found[gi] = (union, len(union))
                        if kind == "g":
                            del found[j]
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                return found

    def bound_groups(self, min_size=2, tidal=True, tidal_threshold=1.0 / 3.0):
        """Gravitationally bound groups at the current instant, as a list of BoundGroup (largest first).
        Every returned group satisfies, in its own centre-of-mass frame: each member has
        1/2 |v_i - V|^2 + phi_i < 0 (bound to the rest of the group) and T + W < 0 (the group is bound).
        Seeds come from pairwise two-body binding, then members are peeled off until both conditions hold; the
        stripped bodies are re-linked and processed the same way; finally groups absorb lone bodies / other groups
        whose union is collectively bound (phase-independent membership). Groups are disjoint.
        The influence of bodies outside a group is reported, not included in E: tidal_ratio is the largest
        (m_k / M)(r_max / D_k)^3 over outside bodies k, and `perturbed` is set when it exceeds tidal_threshold
        (1/3 is the Hill-sphere criterion: the group extent r_max reaches outside body k's Hill sphere)."""
        min_size = max(2, int(min_size))
        live = self._live()
        if live.numel() < 2:
            return []
        ids_np = self.ids.cpu().numpy()
        queue = deque(c for c in self._components(live) if len(c) >= 2)
        found = []
        while queue:
            s = queue.popleft()
            alive = self._unbind(torch.as_tensor(s, device=self.dev))
            kept, ejected = s[alive], s[~alive]
            if len(kept) >= 2:
                found.append((kept, len(s)))
            if len(ejected) >= 2:
                queue.extend(c for c in self._components(torch.as_tensor(ejected, device=self.dev)) if len(c) >= 2)
        found = self._merge_objects(found)
        groups = [self._describe(kept, n_linked, live, ids_np, tidal, tidal_threshold)
                  for kept, n_linked in found if len(kept) >= min_size]
        groups.sort(key=lambda g: (-g.size, g.ids[0]))
        return groups

    def calculate_dynamics(self, min_size=2, show=15, verbose=True, tidal=True, tidal_threshold=1.0 / 3.0):
        """Report the gravitationally bound groups (see bound_groups) and return them as BoundGroup objects."""
        groups = self.bound_groups(min_size=min_size, tidal=tidal, tidal_threshold=tidal_threshold)
        if verbose:
            n_live = int((self.mass > 0).sum())
            in_g = sum(g.size for g in groups)
            print(f"Bound groups  t={self.t:.4f} yr  live bodies={n_live}: "
                  f"{len(groups)} group(s) holding {in_g} bodies, {n_live - in_g} in no group")
            if VERBOSE:                                    # extra O(N^2) seed analysis only on request
                seeds = self.pairwise_linked_groups(min_size=2)
                n_seed = sum(len(s) for s in seeds)
                print(f"  seeds (pairwise-bound components, >= 2 bodies): {len(seeds)}, {n_seed} bodies; "
                      f"{n_seed - in_g} seed member(s) failed the collective test")
            if groups:
                print(f"  {'#':>3} {'size':>6} {'mass':>10} {'E_group':>12} {'2T/|W|':>7} {'r_half':>9} "
                      f"{'tidal':>9}  stripped  members")
            for n, g in enumerate(groups[:show], 1):
                body = str(g.ids) if g.size <= 8 else str(g.ids[:8])[:-1] + f", ... +{g.size - 8}]"
                tid = "n/a" if g.tidal_ratio != g.tidal_ratio else f"{g.tidal_ratio:.2e}"
                print(f"  {n:>3} {g.size:>6} {g.mass:>10.4g} {g.energy:>12.5e} {g.virial_ratio:>7.3f} "
                      f"{g.r_half:>9.3e} {tid:>8}{'*' if g.perturbed else ' '} {g.n_stripped:>8}  {body}")
            if len(groups) > show:
                print(f"  ... {len(groups) - show} more group(s)")
            if any(g.perturbed for g in groups):
                print(f"  * tidal ratio > {tidal_threshold:.3g}: the group extends past an outside body's Hill "
                      "sphere, so its boundness is strongly affected by that body")
            if VERBOSE:
                print("  units: M_sun, AU, yr; E, T, W in the group's own COM frame; r_half about the COM")
        return groups

    # ====================================================================== long-term stability
    def _stability_snapshot(self, params, ref_scale=None):
        """Instantaneous structure: bound groups (each assessed), the core, escape test for every other object."""
        live = self._live()
        n = live.numel()
        if n == 0:
            return StabilityReport(self.t, 0, [], [], {}, [], {}, float("nan"), float("nan"))
        ids_all = self.ids[live].cpu().numpy()
        P = self.pos[live].double().cpu().numpy()
        V = self.vel[live].double().cpu().numpy()
        M = self.mass[live].double().cpu().numpy()
        row = {int(i): r for r, i in enumerate(ids_all)}
        Mt = float(M.sum())
        Xcm = (M[:, None] * P).sum(0) / Mt
        Vcm = (M[:, None] * V).sum(0) / Mt
        rms = math.sqrt(float(((P - Xcm) ** 2).sum(1).mean()))
        t_dyn = math.sqrt(max(rms, 1e-9) ** 3 / (G * Mt))

        groups = self.bound_groups(min_size=2, tidal=True) if n >= 2 else []
        objects, grouped = [], set()
        for g in groups:
            rows = np.array([row[i] for i in g.ids])
            grouped.update(g.ids)
            gs = _assess_group(g.ids, M[rows], P[rows], V[rows], params, g.perturbed, g.tidal_source)
            objects.append(dict(ids=tuple(g.ids), rows=rows, m=g.mass, X=np.array(g.com_pos),
                                V=np.array(g.com_vel), rmax=g.r_max, gs=gs))
        gstab = [o["gs"] for o in objects]
        for r, i in enumerate(ids_all):
            if int(i) not in grouped:
                objects.append(dict(ids=(int(i),), rows=np.array([r]), m=float(M[r]), X=P[r], V=V[r],
                                    rmax=0.0, gs=None))
        core = max(objects, key=lambda o: (o["m"], len(o["ids"]), -o["ids"][0]))

        fates, escaping, unresolved = {}, {}, []
        label = {"stable": "bound-stable", "marginal": "bound-marginal", "unstable": "bound-unstable",
                 "unresolved": "bound-unresolved"}
        for i in core["ids"]:
            fates[i] = label[core["gs"].verdict] if core["gs"] is not None else "core"
        for o in objects:
            if o is core:
                continue
            keep = np.ones(n, bool)
            keep[o["rows"]] = False
            d = o["X"] - P[keep]
            phi = -G * float((M[keep] / np.sqrt((d * d).sum(1) + EPS2)).sum())
            e_o = 0.5 * float(((o["V"] - Vcm) ** 2).sum()) + phi          # energy per unit mass, system COM frame
            rel, vrel = o["X"] - core["X"], o["V"] - core["V"]
            D = math.hypot(rel[0], rel[1])
            scale = max(core["rmax"], ref_scale or 0.0)
            far = (D > params.escape_factor * scale) if scale > 0 else True
            if e_o > 0.0 and float(rel @ vrel) > 0.0 and far:
                escaping[o["ids"]] = dict(D=D, e=e_o, v=math.hypot(vrel[0], vrel[1]))
                st = "escaping"
            else:
                unresolved.append(o["ids"])
                st = "unresolved"
            for i in o["ids"]:
                fates[i] = st
        p = [g.p_max for g in gstab if g.p_max == g.p_max]
        return StabilityReport(self.t, n, list(core["ids"]), gstab, escaping, unresolved, fates,
                               max(p) if p else float("nan"), t_dyn, energy=float(self.energy()), r_rms=rms)

    def stability_report(self, monitor=None, params=None, verbose=True):
        """Is the system settled? Every body must be in a bound structure that passes an established stability
        criterion, or be on a confirmed escape trajectory, and the structure must have been unchanged for a long
        window (>= params.min_dyn dynamical times and >= params.min_periods of the longest bound orbit).
        Pass a StabilityMonitor that has been fed with monitor.record(sim) during the run to judge persistence;
        without one the best possible verdict is PENDING. Returns a StabilityReport (.verdict, .settled, .fates)."""
        params = params or (monitor.params if monitor is not None else StabilityParams())
        rep = _finalize_report(self._stability_snapshot(params, monitor.r0 if monitor is not None else None),
                               params, monitor)
        if verbose:
            print(rep.summary() if VERBOSE else rep.brief())
        return rep

    def diagnostics(self):
        """Cheap global summary (uses the potential left behind by the last force pass)."""
        ke = float(self.kinetic_energy_per_body().sum())
        e = float(self.energy())
        pe = e - ke
        return dict(t=self.t, kinetic=ke, potential=pe, total=e,
                    virial=(2 * ke / abs(pe)) if pe != 0 else float("nan"),
                    angular_momentum_z=float(self.angular_momentum().sum()),
                    alive=int((self.mass > 0).sum()))


# ====================================================================================== helpers
def random_scene(n, seed=0):
    """Stress-test cloud (not physically meaningful): n bodies in a disc, ~1.3 bodies per AU^2."""
    rng = np.random.default_rng(seed)
    R = math.sqrt(n / (1.3 * math.pi))
    r = R * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    pos = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return pos, rng.normal(0, 0.5, (n, 2)), rng.uniform(0.01, 0.1, n), np.full(n, 2.0)


def pick_device():
    forced = os.environ.get("NBODY_CUDA_DEVICE")
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{forced}" if forced else "cuda:0")
        print(f"[compute] CUDA device: {torch.cuda.get_device_name(dev)}")
        return dev
    print("[compute] WARNING: CUDA not available -> CPU fallback (slow).")
    return torch.device("cpu")


def set_windows_gpu_preference():
    if sys.platform != "win32":
        print("Only needed on Windows. On Linux hybrid laptops: "
              "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python gpu_main.py")
        return
    import winreg
    exes = {sys.executable, getattr(sys, "_base_executable", sys.executable)}
    for e in list(exes):
        w = os.path.join(os.path.dirname(e), "pythonw.exe")
        if os.path.exists(w):
            exes.add(w)
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\DirectX\UserGpuPreferences")
    for e in sorted(exes):
        winreg.SetValueEx(key, e, 0, winreg.REG_SZ, "GpuPreference=2;")
        print("High-performance GPU preference set for:", e)
    print("Restart the program (and the terminal/editor that launched it).")


def run_bench(device, dtype, force_dtype, use_compile, kernel, sizes=(1000, 2000, 5000, 10000, 20000)):
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def timeit(fn, reps):
        fn(); sync()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        return (time.perf_counter() - t) / reps * 1e3

    print(f"dtype={dtype}, force_dtype={force_dtype or dtype}, compile={use_compile}, kernel={kernel}")
    print(f"{'N':>7} | {'force eager':>11} | {'force fused':>11} {'(G pair/s)':>10} | {'merge eager':>11} | "
          f"{'merge fused':>11} | {'stats+dt':>8} | {'full step':>9} | {'fetch':>6}   (ms)")
    for n in sizes:
        pos, vel, mass, rad = random_scene(n)
        try:
            sim = GPUNBody(pos, vel, mass, rad, device, dtype, force_dtype, use_compile=use_compile, kernel=kernel)
            reps = 3 if n >= 10000 else 10
            fe = timeit(lambda: sim._accel_eager(sim.pos, sim.mass), reps)
            ff = timeit(lambda: sim._accel_fused(sim.pos, sim.mass), reps) if sim._fused else float("nan")
            alive = sim.mass > 0
            order = torch.sort(sim.mass, stable=True).indices
            rank = torch.empty_like(order); rank[order] = torch.arange(n, device=device)
            r_au = sim.rad * AU_p_Pix
            me = timeit(lambda: sim._best_rank_eager(sim.pos, r_au, alive, rank), reps)
            mf = timeit(lambda: sim._best_rank_fused(sim.pos, r_au, sim.mass, rank), reps) if sim._fused else float("nan")
            cl = sim._accel_and_closest()[1]
            sd = timeit(lambda: compute_dt(sim._closest_stats(cl), sim.dt, sim.dtp), reps)
            st = timeit(sim.step, reps)
            fr = timeit(lambda: sim.fetch_render(0.0, 0.0), reps)
            print(f"{n:>7} | {fe:11.2f} | {ff:11.2f} {n * n / (ff * 1e-3) / 1e9 if ff == ff else float('nan'):10.2f} | "
                  f"{me:11.2f} | {mf:11.2f} | {sd:8.2f} | {st:9.2f} | {fr:6.2f}")
        except RuntimeError as e:
            print(f"{n:>7} | failed: {str(e).splitlines()[0]}")
            break
        del sim
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _np_pair_E(P, V, M, a, b):
    """Two-body energy in the pair's COM frame, softened like the integrator (independent NumPy reference)."""
    r = math.sqrt(float(((P[a] - P[b]) ** 2).sum()) + EPS2)
    v2 = float(((V[a] - V[b]) ** 2).sum())
    return 0.5 * M[a] * M[b] / (M[a] + M[b]) * v2 - G * M[a] * M[b] / r


def _ref_components(P, V, M, subset):
    seen, out = set(), []
    for i in subset:
        if i in seen:
            continue
        st, grp = [i], {i}
        seen.add(i)
        while st:
            a = st.pop()
            for b in subset:
                if b not in seen and _np_pair_E(P, V, M, a, b) < 0:
                    seen.add(b)
                    st.append(b)
                    grp.add(b)
        out.append(sorted(grp))
    return out


def _ref_member_energies(P, V, M, mem):
    """(e_i, T_com, W) of the member set from scratch: the definition, no incremental updates."""
    p, v, m = P[mem], V[mem], M[mem]
    Mt = m.sum()
    Vc = (m[:, None] * v).sum(0) / Mt
    d = p[:, None, :] - p[None, :, :]
    U = m[None, :] / np.sqrt((d * d).sum(-1) + EPS2)
    np.fill_diagonal(U, 0.0)
    phi = -G * U.sum(1)
    ke = 0.5 * ((v - Vc) ** 2).sum(1)
    return ke + phi, float((m * ke).sum()), float(0.5 * (m * phi).sum())


def _ref_merge(P, V, M, groups, live, max_try=8):
    """Reference merge stage: from-scratch union energies (no analytic decomposition)."""
    groups = [list(g) for g in groups]
    while True:
        grouped = {i for g in groups for i in g}
        singles = [int(i) for i in live if int(i) not in grouped]
        order = sorted(range(len(groups)), key=lambda i: -float(M[groups[i]].sum()))
        merged = False
        for gi in order:
            cands = []
            for j in singles:
                _, T, W = _ref_member_energies(P, V, M, sorted(groups[gi] + [j]))
                if T + W < 0:
                    cands.append((T + W, "s", j))
            for gj in range(len(groups)):
                if gj != gi:
                    _, T, W = _ref_member_energies(P, V, M, sorted(groups[gi] + groups[gj]))
                    if T + W < 0:
                        cands.append((T + W, "g", gj))
            cands.sort(key=lambda c: c[0])
            for _, kind, j in cands[:max_try]:
                u = sorted(groups[gi] + ([j] if kind == "s" else groups[j]))
                e, T, W = _ref_member_energies(P, V, M, u)
                if e.max() < 0 and T + W < 0:
                    groups[gi] = u
                    if kind == "g":
                        del groups[j]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return groups


def _ref_bound_groups(P, V, M, live):
    """Pure-Python version of the algorithm: seed, peel the most unbound member one at a time, re-link the rest."""
    queue = [c for c in _ref_components(P, V, M, list(live)) if len(c) >= 2]
    groups = []
    while queue:
        s = queue.pop(0)
        mem = list(s)
        while len(mem) >= 2:
            e, T, W = _ref_member_energies(P, V, M, mem)
            if e.max() < 0 and T + W < 0:
                break
            mem.pop(int(np.argmax(e)))
        if len(mem) >= 2:
            groups.append(tuple(mem))
        ejected = [x for x in s if x not in set(mem)]
        if len(ejected) >= 2:
            queue.extend(c for c in _ref_components(P, V, M, ejected) if len(c) >= 2)
    return [tuple(g) for g in _ref_merge(P, V, M, groups, live)]


def run_group_checks(device):
    """bound_groups() against an independent reference, its defining invariants, and three analytic cases."""
    ok = True

    def report(good, text):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK  ' if good else 'FAIL'} {text}")

    def check_scene(label, sim):
        P, V, M = (x.cpu().numpy() for x in (sim.pos, sim.vel, sim.mass))
        live = np.flatnonzero(M > 0)
        got = sim.bound_groups(min_size=2, tidal=False)
        ref = _ref_bound_groups(P, V, M, live)
        seeds = [c for c in _ref_components(P, V, M, list(live)) if len(c) >= 2]
        same = {tuple(g.ids) for g in got} == {tuple(g) for g in ref}
        report(same, f"{label}: bound_groups == reference ({len(got)} groups from {len(seeds)} seeds, "
                     f"{sum(g.n_stripped > 0 for g in got)} lost members to peeling, "
                     f"seed members {sum(len(c) for c in seeds)} -> grouped {sum(g.size for g in got)})")
        # 2) invariants, recomputed from scratch for each returned group
        worst_e, worst_E, derr = -np.inf, -np.inf, 0.0
        for g in got:
            e, T, W = _ref_member_energies(P, V, M, g.ids)
            worst_e, worst_E = max(worst_e, float(e.max())), max(worst_E, T + W)
            derr = max(derr, abs(g.energy - (T + W)) / abs(T + W))
        allids = [i for g in got for i in g.ids]
        report(worst_e < 0 and worst_E < 0 and derr < 1e-9 and len(allids) == len(set(allids)),
               f"{label}: every member e_i < 0 (max {worst_e:.2e}), every group E < 0 (max {worst_E:.2e}), "
               f"groups disjoint, reported energy matches (err {derr:.1e})")

    # 1) hot clustered scenes (the peeling really removes bodies here): same groups as the reference
    for seed in range(3):
        rng = np.random.default_rng(seed)
        k, per = 6, 28
        cen = rng.uniform(-30, 30, (k, 2))
        pos = np.concatenate([c + rng.normal(0, 0.5, (per, 2)) for c in cen])
        vel = np.concatenate([rng.normal(0, 3.0 + 4.0 * i, (per, 2)) for i in range(k)])
        m = rng.uniform(0.5, 2.0, k * per)
        sim = GPUNBody(pos, vel, m, np.full(len(m), 2.0), device, torch.float64, kernel="eager")
        sim.mass[:8] = 0.0
        sim.vel[:8] = 0.0
        sim.pos[:8] = 0.0
        check_scene(f"clustered {seed}", sim)

    # 1b) hot random clouds: seeds are large chains, so members really are stripped and re-linked
    for n, vs in ((120, 45.0), (200, 45.0), (200, 70.0)):
        pos, vel, m, rad = random_scene(n, seed=5)
        sim = GPUNBody(pos, vel * vs, m * 10, rad, device, torch.float64, kernel="eager")
        check_scene(f"hot cloud N={n} v x{vs:g}", sim)

    # 3) two bodies: group energy must equal the analytic two-body energy
    pos = np.array([[0.0, 0.0], [1.5, 0.0]])
    vel = np.array([[0.0, 0.0], [0.0, 10.0]])
    sim = GPUNBody(pos, vel, np.array([1.0, 3.0]), np.full(2, 2.0), device, torch.float64, kernel="eager")
    mu = 3.0 / 4.0
    r = math.sqrt(1.5 ** 2 + EPS2)
    analytic = 0.5 * mu * 100.0 - G * 3.0 / r
    g = sim.bound_groups(tidal=False)
    E_pair = float(sim.pairwise_energy()[0][0, 1])
    report(len(g) == 1 and abs(g[0].energy - analytic) < 1e-9 * abs(analytic) and abs(E_pair - analytic) < 1e-6 * abs(analytic),
           f"two bodies: E_group = {g[0].energy:.6f}, analytic {analytic:.6f}, pairwise_energy {E_pair:.6f}")

    # 4) the case a pure pairwise graph gets wrong: A-B and B-C pairwise bound, A+B+C not bound
    d, u2 = 1.0, 3.5 * G
    u = math.sqrt(u2)
    pos = np.array([[-d, 0.0], [0.0, 0.0], [d, 0.0]])
    vel = np.array([[-u, 0.0], [0.0, 0.0], [u, 0.0]])
    sim = GPUNBody(pos, vel, np.ones(3), np.full(3, 2.0), device, torch.float64, kernel="eager")
    P, V, M = (x.cpu().numpy() for x in (sim.pos, sim.vel, sim.mass))
    e3, T3, W3 = _ref_member_energies(P, V, M, [0, 1, 2])
    linked = sim.pairwise_linked_groups(min_size=2)
    g = sim.bound_groups(tidal=False)
    report(linked == [[0, 1, 2]] and T3 + W3 > 0 and len(g) == 1 and g[0].size == 2 and 1 in g[0].ids and g[0].energy < 0,
           f"chain A-B-C: pairwise graph links {linked}, but E(A+B+C) = {T3 + W3:+.2f} > 0; "
           f"bound_groups gives {[x.ids for x in g]} with E = {g[0].energy:.2f} < 0")

    # 5) tidal ratio: a bound pair with a heavy outsider far away (small ratio) and close by (Hill criterion tripped)
    for D, expect_flag in ((50.0, False), (1.0, True)):
        pos = np.array([[-0.5, 0.0], [0.5, 0.0], [D, 0.0]])
        vel = np.array([[0.0, 3.0], [0.0, -3.0], [0.0, 100.0]])
        sim = GPUNBody(pos, vel, np.array([1.0, 1.0, 10.0]), np.full(3, 2.0), device, torch.float64, kernel="eager")
        g = sim.bound_groups()
        want = (10.0 / 2.0) * (0.5 / D) ** 3
        good = (len(g) == 1 and g[0].ids == [0, 1] and abs(g[0].tidal_ratio - want) < 1e-9 * want
                and g[0].tidal_source == 2 and g[0].perturbed == expect_flag)
        report(good, f"tidal ratio, outsider at {D:g} AU: {g[0].tidal_ratio:.3e} (expected {want:.3e}), "
                     f"perturbed={g[0].perturbed}")
    return ok


def run_stability_checks():
    """Long-term stability: analytic criteria, then short integrations with the monitor (always on CPU, tiny N)."""
    dev = torch.device("cpu")
    ok = True

    def report(good, text):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK  ' if good else 'FAIL'} {text}")

    def mk(P, V, M):
        return GPUNBody(np.array(P, float), np.array(V, float), np.array(M, float), np.full(len(M), 2.0), dev,
                        torch.float64, kernel="eager")

    def triple(a_out, retro=False, a_in=0.5):          # inner circular binary (1, 1) + outer body (1), COM at rest
        X1, X2 = -a_out / 3.0, 2.0 * a_out / 3.0
        vo = math.sqrt(G * 3.0 / a_out) * (-1 if retro else 1)
        V1, V2 = -vo / 3.0, 2.0 * vo / 3.0
        vin = math.sqrt(G * 2.0 / a_in)
        return ([[X1 - a_in / 2, 0], [X1 + a_in / 2, 0], [X2, 0]],
                [[0, V1 - vin / 2], [0, V1 + vin / 2], [0, V2]], [1.0, 1.0, 1.0])

    def advance(sim, mon, t_end, dt, every):
        sim.fixed_dt_value = dt
        sim.set_fixed_dt(True)
        nxt = sim.t
        while sim.t < t_end:
            sim.step()
            if sim.t >= nxt:
                mon.record(sim)
                nxt += every

    # 1) Kepler elements against an independent construction (arbitrary phase, rotated frame)
    a0, e0, mA, mB, f, rot = 2.0, 0.3, 1.0, 0.5, 1.1, 0.7
    mu = G * (mA + mB)
    p = a0 * (1 - e0 ** 2)
    r = p / (1 + e0 * math.cos(f))
    vr, vt = math.sqrt(mu / p) * e0 * math.sin(f), math.sqrt(mu / p) * (1 + e0 * math.cos(f))
    x = np.array([r * math.cos(f), r * math.sin(f)])
    v = np.array([vr * math.cos(f) - vt * math.sin(f), vr * math.sin(f) + vt * math.cos(f)])
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    o = _kepler(mA, mB, R @ x, R @ v)
    err = max(abs(o["a"] - a0), abs(o["e"] - e0), abs(o["period"] - 2 * math.pi * math.sqrt(a0 ** 3 / mu)))
    report(err < 1e-12, f"osculating elements (a, e, P) recovered from a rotated state at arbitrary phase (err {err:.1e})")

    # 2) hierarchical triples with a known Mardling-Aarseth margin (q_out = 0.5, e_out = 0)
    for a_out, retro, want_v in ((2.5, False, "stable"), (1.2, False, "unstable"), (2.5, True, "stable")):
        gs = mk(*triple(a_out, retro)).stability_report(verbose=False).groups[0]
        want = (a_out / 0.5) / (2.8 * 1.5 ** 0.4 * (0.7 if retro else 1.0))
        report(gs.kind == "hierarchical" and gs.verdict == want_v and abs(gs.margin - want) < 1e-6,
               f"triple a_out={a_out:g} {'retrograde' if retro else 'prograde'}: margin {gs.margin:.4f} "
               f"(expected {want:.4f}) -> {gs.verdict}")

    # 3) planetary spacing in mutual Hill radii: star + two 1e-3 planets, circular
    for a2, want_v in ((1.8, "stable"), (1.3, "unstable")):
        m = 1e-3
        sim = mk([[0, 0], [1.0, 0], [-a2, 0]],
                 [[0, 0], [0, math.sqrt(G * (1 + m))], [0, -math.sqrt(G * (1 + m) / a2)]], [1.0, m, m])
        gs = sim.stability_report(verbose=False).groups[0]
        rh = (2 * m / 3) ** (1 / 3) * 0.5 * (1 + a2)
        want = ((a2 - 1.0) / rh) / (2 * math.sqrt(3))
        report(gs.kind == "planetary" and gs.verdict == want_v and abs(gs.margin - want) < 1e-6,
               f"planets at 1 and {a2:g} AU: margin {gs.margin:.4f} (expected {want:.4f}) -> {gs.verdict}")

    # 4) equilateral triple at rest (the default demo): not a stable hierarchy
    sim = mk([[0, 5.0], [4.3301, -2.5], [-4.3301, -2.5]], [[0, 0]] * 3, [2, 2, 2])
    rep = sim.stability_report(verbose=False)
    report(rep.verdict == "NOT SETTLED" and rep.groups[0].verdict == "unstable",
           f"equilateral triple at rest -> {rep.verdict} ({rep.groups[0].verdict}, margin {rep.groups[0].margin:.2g})")

    # 5) group membership must not depend on orbital phase (stable triple, many phases)
    sim = mk(*triple(2.5))
    mon = StabilityMonitor(StabilityParams(min_periods=10, min_dyn=2))
    advance(sim, mon, 30.0, 0.004, 0.25)
    parts = {r.sig[1] for r in mon.records}
    report(len(parts) == 1 and len(next(iter(parts))) == 1, f"stable triple: one group at all {len(mon.records)} snapshots")
    rep = sim.stability_report(monitor=mon, verbose=False)
    report(rep.verdict == "SETTLED" and rep.energy_drift < 1e-3,
           f"stable triple after 30 yr: {rep.verdict} (quiet {rep.quiet_span:.1f} of {rep.required_span:.1f} yr, "
           f"dE/E {rep.energy_drift:.1e})")
    early = StabilityMonitor(StabilityParams(min_periods=10, min_dyn=2))
    sim0 = mk(*triple(2.5))
    advance(sim0, early, 3.0, 0.004, 0.25)
    rep0 = sim0.stability_report(monitor=early, verbose=False)
    report(rep0.verdict == "PENDING", f"same system after only 3 yr: {rep0.verdict} (quiet window too short, as it must be)")

    # 6) binary + fast outbound body: 'escaping' at first, 'escaped' once the window has passed
    vc = math.sqrt(G * 4 / 0.5)
    sim = mk([[-0.25, 0], [0.25, 0], [10, 0]], [[0, -vc / 2], [0, vc / 2], [20, 0]], [2, 2, 0.1])
    r0 = sim.stability_report(verbose=False)
    mon = StabilityMonitor(StabilityParams(min_periods=10, min_dyn=2))
    advance(sim, mon, 12.0, 0.002, 0.25)
    rep = sim.stability_report(monitor=mon, verbose=False)
    report(r0.fates[2] == "escaping" and rep.fates[2] == "escaped" and rep.verdict == "SETTLED",
           f"binary + outbound body: {r0.fates[2]} at t=0 -> {rep.fates[2]}, {rep.verdict} at t={rep.t:.1f} yr")

    # 7) a run whose energy is not conserved must not be certified (deliberately coarse step)
    sim = mk(*triple(2.5))
    mon = StabilityMonitor(StabilityParams(min_periods=10, min_dyn=2))
    advance(sim, mon, 30.0, 0.04, 0.25)
    rep = sim.stability_report(monitor=mon, verbose=False)
    report(rep.verdict == "NOT SETTLED" and rep.energy_drift > 1e-3,
           f"coarse time step: {rep.verdict}, dE/E = {rep.energy_drift:.1e} flagged as untrustworthy")
    return ok


def run_analysis_check(device, n=400, dead=40, primary=100):
    """Compare every body-state diagnostic with a NumPy port of the original CPU formulas (float64)."""
    pos, vel, mass, rad = random_scene(n)
    sim = GPUNBody(pos, vel, mass, rad, device, torch.float64, kernel="eager")
    sim.mass[:dead] = 0.0                                   # emulate merged-away bodies
    sim.vel[:dead] = 0.0
    sim.pos[:dead] = 0.0
    live = np.flatnonzero(sim.mass.cpu().numpy() > 0)

    P, V, M = (x.cpu().numpy() for x in (sim.pos, sim.vel, sim.mass))
    dist = lambda A: np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
    r_ref, v_ref = dist(P), dist(V)
    act = (M[:, None] != 0) & (M[None, :] != 0)
    np.fill_diagonal(act, False)
    mul, add = M[:, None] * M[None, :], M[:, None] + M[None, :]

    orb = np.full_like(r_ref, np.nan)
    pair = np.full_like(r_ref, np.nan)
    pair_s = np.full_like(r_ref, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        orb[act] = 0.5 * v_ref[act] ** 2 - G * add[act] / r_ref[act]
        pair[act] = 0.5 * (mul[act] / add[act]) * v_ref[act] ** 2 - G * mul[act] / r_ref[act]
        pair_s[act] = (0.5 * (mul[act] / add[act]) * v_ref[act] ** 2
                       - G * mul[act] / np.sqrt(r_ref[act] ** 2 + EPS2))      # softened, as the seed stage uses
        r_pr = np.linalg.norm(P - P[primary], axis=1)
        tid = (M[None, :] / M[primary]) * (r_pr[:, None] / r_pr[None, :]) ** 3
    tid[primary] = np.nan
    tid[:, primary] = np.nan
    np.fill_diagonal(tid, np.nan)
    ang = (P[:, 0] * V[:, 1] - P[:, 1] * V[:, 0]) * M          # z-component (np.cross on 2-D is removed in NumPy 2)
    ke = 0.5 * M * np.sum(V ** 2, axis=1)
    pe = np.zeros_like(M)
    d = P[live][:, None, :] - P[live][None, :, :]
    rr = np.sqrt(np.einsum("ijk,ijk->ij", d, d) + EPS2)
    U = -G * M[live][:, None] * M[live][None, :] / rr
    np.fill_diagonal(U, 0.0)
    pe[live] = 0.5 * U.sum(1)
    N = pair.shape[0]
    seen, ref_groups = [False] * N, []                      # the original DFS, restricted to live bodies
    for i in live:
        if seen[i]:
            continue
        stack, grp = [i], {i}
        seen[i] = True
        while stack:
            a = stack.pop()
            for b in live:
                if not seen[b] and pair_s[a, b] < 0:
                    seen[b] = True
                    stack.append(b)
                    grp.add(b)
        ref_groups.append(sorted(int(x) for x in grp))

    def rel(a, b):
        sc = np.nanmax(np.abs(b)) or 1.0
        return float(np.nanmax(np.abs(a - b)) / sc)

    sub = np.ix_(live, live)
    E_orb, ids = sim.orbital_energy()
    E_pair, _ = sim.pairwise_energy()
    E_tid, _ = sim.tidal_perturbation(primary)
    assert ids.cpu().numpy().tolist() == live.tolist()
    # tidal entries can be huge (r_j -> 0 near the primary); compare relative per element instead
    t_a, t_b = E_tid.cpu().numpy(), tid[sub]
    tid_err = float(np.nanmax(np.abs(t_a - t_b) / np.maximum(np.abs(t_b), 1e-300)))
    got_groups = sim.pairwise_linked_groups(min_size=1)
    ref_sorted = sorted(ref_groups, key=lambda g: (-len(g), g[0]))
    ke_t, pe_t = sim.kinetic_energy_per_body().cpu().numpy(), sim.potential_energy_per_body().cpu().numpy()
    E_ref = float(sim.energy_reference())

    rows = [("orbital_energy", rel(E_orb.cpu().numpy(), orb[sub]), 1e-9),
            ("pairwise_energy", rel(E_pair.cpu().numpy(), pair[sub]), 1e-9),
            ("tidal_perturbation (per elem)", tid_err, 1e-9),
            ("angular_momentum", rel(sim.angular_momentum().cpu().numpy(), ang), 1e-12),
            ("kinetic_energy_per_body", rel(ke_t, ke), 1e-12),
            ("potential_energy_per_body", rel(pe_t, pe), 1e-9),
            ("sum(total_energy_per_body) vs energy_reference",
             abs(float(sim.total_energy_per_body().sum()) - E_ref) / abs(E_ref), 1e-9)]
    ok = True
    for name, err, tol in rows:
        good = err < tol
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {name:<48} rel err {err:.2e}")
    same = got_groups == ref_sorted
    ok &= same
    multi = [g for g in ref_sorted if len(g) > 1]
    print(f"  {'OK  ' if same else 'FAIL'} pairwise_linked_groups (seeds) match the CPU DFS "
          f"({len(multi)} multi-body group(s), largest {len(multi[0]) if multi else 0}, "
          f"{len(ref_sorted)} components total)")
    print("bound-group analysis (collective energy criterion):")
    ok &= run_group_checks(device)
    print("long-term stability analysis (criteria + monitored integrations, CPU):")
    ok &= run_stability_checks()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


# ====================================================================================== front end
class Celestial(arcade.SpriteCircle):
    def __init__(self, Name, index, radius, color):
        super().__init__(int(radius), color)
        self.name = Name
        self.index = index
        self.r_px = float(radius)
        self.color = color

    def update_texture(self, r_px, color):
        self.r_px = float(r_px)
        r = max(1, int(round(r_px)))
        self.texture = arcade.make_circle_texture(r * 2, color)
        self.width = r * 2
        self.height = r * 2


def addCelestial(spriteList=None, sprites_by_id=None, Name="", radius=1, position=(0, 0), mass=0,
                 velocity=(0, 0), acceleration=(0, 0), color=arcade.color.WHITE):
    """Same call signature as before (first two args ignored; sprites are built in setup()).
    `Name` is what the NAME labels show (default "Planet<index>")."""
    _init_name.append(Name or f"Planet{len(_init_pos)}")
    _init_pos.append(position)
    _init_vel.append(velocity)
    _init_mass.append(mass)
    _init_radius.append(radius)
    _init_color.append(color)


def build_scene(random_n=None, seed=0):
    """THE place that defines the initial conditions. The window and nbody_headless.py both call this, so a
    GUI run and a headless run start from exactly the same state. Edit the addCelestial calls below.
    Returns (pos, vel, mass, radius_px, colors, names), already shifted to the centre-of-mass frame."""
    for lst in (_init_pos, _init_vel, _init_mass, _init_radius, _init_color, _init_name):
        lst.clear()
    n_rand = RANDOM_N if random_n is None else random_n
    if n_rand:
        pos, vel, mass, rad = random_scene(n_rand, seed)
        colors = [arcade.color.WHITE] * n_rand
        names = [f"P{i}" for i in range(n_rand)]
    else:
        addCelestial(Name="Planet0", radius=5, position=(0, 5), mass=2, velocity=(0, 0))
        addCelestial(Name="Planet1", radius=5, position=(4.3301, -2.5), mass=2, velocity=(0, 0))
        addCelestial(Name="Planet2", radius=5, position=(-4.3301, -2.5), mass=2, velocity=(0, 0))
        pos = np.array(_init_pos, dtype=np.float64)
        vel = np.array(_init_vel, dtype=np.float64)
        mass = np.array(_init_mass, dtype=np.float64)
        rad = np.array(_init_radius, dtype=np.float64)
        colors, names = list(_init_color), list(_init_name)
    pos = pos - (pos * mass[:, None]).sum(0) / mass.sum()
    vel = vel - (vel * mass[:, None]).sum(0) / mass.sum()
    return pos, vel, mass, rad, colors, names


def merge_lines(events, names, members=None, n_alive=None, max_lines=4):
    """Turn GPUNBody.merge_log events into terminal lines (shared by the window and nbody_headless.py).
    `members` (body id -> ids it consists of) is updated in place when given. Returns a list of strings."""
    lines = []
    for t, lost, won, mass in events:
        by_winner = {}
        for l, w, m in zip(lost, won, mass):
            by_winner.setdefault(w, [m, []])[1].append(l)
        for w, (m, ls) in by_winner.items():
            if members is not None:
                members[w] = members.get(w, [w]) + sum((members.pop(l, [l]) for l in ls), [])
            lines.append(f"[t = {t:10.3f} yr] MERGE  {' + '.join(names[l] for l in ls)} -> absorbed by "
                         f"{names[w]} (mass now {m:.4g})")
    tail = "" if n_alive is None else f"   | alive: {n_alive}"
    if len(lines) <= max_lines:
        return [ln + (tail if k == len(lines) - 1 else "") for k, ln in enumerate(lines)]
    return [f"[t = {events[-1][0]:10.3f} yr] {len(lines)} merges{tail}"]


def _tracker_label(mode, bid, names, members):
    """Label text for body `bid` in label mode 'id' or 'name'; '(+k)' = it has absorbed k other bodies."""
    base = str(bid) if mode == "id" else names[bid]
    k = len(members.get(bid, ())) - 1
    return f"{base} (+{k})" if k > 0 else base


POINT_VS = """
#version 330
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform float zoom;
in vec2 in_pos;
in float in_radius;
out vec2 v_uv;
void main() {
    // one instanced quad (triangle strip, 4 vertices) per body; world units are pixels, so size = radius
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    float r = (in_radius > 0.0) ? max(in_radius, 1.5 / zoom) : 0.0;   // at least ~1.5 px on screen
    v_uv = corner;
    gl_Position = window.projection * window.view * vec4(in_pos + corner * r, 0.0, 1.0);
}
"""
POINT_FS = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
void main() {
    if (dot(v_uv, v_uv) > 1.0) discard;
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

TRAIL_VS = """
#version 430
uniform WindowBlock { mat4 projection; mat4 view; } window;
layout(std430, binding = 0) buffer TrailData { vec2 pts[]; };
uniform int n_bodies;
uniform int n_slots;
uniform int head;
uniform float fade;      // 1 = older trail points fade out, 0 = constant opacity
out vec4 v_color;
const vec3 PAL[7] = vec3[7](vec3(1.0, 0.0, 0.0), vec3(0.2, 0.4, 1.0), vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0),
                            vec3(1.0, 0.0, 1.0), vec3(1.0, 0.75, 0.8), vec3(0.6, 0.2, 0.8));
void main() {
    int seg = gl_VertexID >> 1;
    int endp = gl_VertexID & 1;
    int body = seg % n_bodies;
    int c = seg / n_bodies + endp;
    int slot = (head + c) % n_slots;
    vec2 p = pts[slot * n_bodies + body];
    gl_Position = window.projection * window.view * vec4(p, 0.0, 1.0);
    v_color = vec4(PAL[body % 7], mix(1.0, float(c) / float(n_slots - 1), fade));
}
"""
TRAIL_FS = """
#version 430
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }
"""

GL_PROGRAM_POINT_SIZE = 0x8642


def enable_program_point_size(ctx):
    """arcade's context has no PROGRAM_POINT_SIZE attribute (that was the v3 failure) -> use pyglet's GL."""
    try:
        ctx.enable(ctx.PROGRAM_POINT_SIZE)
        return
    except Exception:
        pass
    from pyglet import gl
    gl.glEnable(getattr(gl, "GL_PROGRAM_POINT_SIZE", GL_PROGRAM_POINT_SIZE))


class myEnvironment(arcade.Window):
    def __init__(self, title="", device=None, dtype=torch.float64, force_dtype=None, use_compile=False,
                 kernel="auto", trail_budget=3_000_000, log_bodies=256, log_every=10, trail_fade=False,
                 group_every=0, stability_every=-1):
        global global_center_x, global_center_y
        self.device_t, self.dtype, self.force_dtype = device, dtype, force_dtype
        self.use_compile, self.kernel = use_compile, kernel
        self.trail_budget, self.log_bodies, self.log_every = trail_budget, log_bodies, log_every
        self.group_every = group_every
        self.stability_every = stability_every
        self.is_fullscreen = False
        self.trail_fade = trail_fade

        screen_w, screen_h = arcade.get_display_size()
        self.window_w = int(screen_w * 0.8)
        self.window_h = int(screen_h * 0.73)
        super().__init__(self.window_w, self.window_h, title, vsync=False)
        self.set_location(0, 38)
        global_center_x = self.center_x
        global_center_y = self.center_y

        try:
            renderer = self.context.get_info().get_renderer()
            say(f"[render]  OpenGL renderer: {renderer}")
            if "nvidia" not in renderer.lower() and device is not None and device.type == "cuda":
                print("[render]  NOTE: drawing is NOT on the NVIDIA GPU (physics still is). Use "
                      "--set-high-perf-gpu or set python.exe to 'High performance' in Windows graphics settings.")
        except Exception as e:  # pragma: no cover
            print("[render]  could not query OpenGL renderer:", e)

    # ---------------------------------------------------------------- setup
    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)
        self.camera = arcade.Camera2D()
        self.paused = False
        self.live_print = False           # P / button: one status line per real second on the terminal
        self._last_print = 0.0

        # time-based stepping: advance a constant amount of SIMULATED time per real second
        self.sim_speed = 0.5              # simulated years per real second (UP/DOWN or buttons)
        self.step_budget_ms = 8.0         # max wall time spent stepping per frame (LEFT/RIGHT to change)
        self.t_goal = 0.0
        self.last_steps = 0
        self.achieved_speed = 0.0
        self.show_hud = True

        self.sprite_mode_max = 300
        self.dead_fraction = 0.05
        self.trail_every = 2              # frames between trail points
        self.planets = arcade.SpriteList()

        pos, vel, mass, rad, colors, names = build_scene()
        self.n0 = n = len(mass)
        self.names = names
        self.members = {i: [i] for i in range(n)}   # body id -> ids it consists of (itself + everything it absorbed)
        self.label_mode = "off"                     # off | id | name
        self.label_max = 150                        # labels are drawn one by one: cap them
        self._labels = {}                           # body id -> arcade.Text (pooled)
        self._last_buf = None
        self.report = None                          # latest finalised StabilityReport (None = not computed yet)

        self.sim = GPUNBody(pos, vel, mass, rad, self.device_t, self.dtype, self.force_dtype,
                            use_compile=self.use_compile, kernel=self.kernel)
        self.t_goal = self.sim.t

        self.use_sprites = n <= self.sprite_mode_max
        self.sprite_of = {}
        self.pt_ok = False
        self._n_draw = 0
        if self.use_sprites:
            for i in range(n):
                s = Celestial(names[i], i, rad[i], colors[i])
                self.planets.append(s)
                self.sprite_of[i] = s
        else:
            self._init_point_renderer(n)

        self.frame_count = 0
        self.log_f = open("simulation_data.csv", "w")
        self.log_f.write("time,object_id,x,y,vx,vy,mass\n")
        self.energy_rows = []
        self.group_rows = []
        self.E0 = float(self.sim.energy())
        # long-term stability: snapshots of the structure so a verdict can demand a long quiet window
        every = self.stability_every if self.stability_every >= 0 else (30 if n <= 1000 else 0)
        self.stability_every = every
        self.monitor = StabilityMonitor() if every > 0 else None

        fd = self.sim.force_dtype
        print(f"[setup] N={n} | draw: {'sprites' if self.use_sprites else 'points'} | forces: "
              f"{self.sim.backend_name}, {'fp32' if fd == torch.float32 else 'fp64'} | stability monitor: "
              f"{'every %d frames' % every if every else 'off (no stop-time estimate; see --stability-every)'}")
        print("[keys]  SPACE pause | L labels | P live print | N snapshot | S stability | B groups | D energy | "
              "UP/DOWN speed | F fullscreen")

        try:
            self.hud = arcade.Text("", 10, 10, arcade.color.WHITE, 13)
        except Exception as e:  # pragma: no cover
            print("[render]  HUD text unavailable:", e)
            self.hud = None

        buf, t, dt, alive = self.sim.fetch_render(global_center_x, global_center_y)
        self._last_buf = buf
        self._init_trails(buf)
        self._show(buf)
        self._build_ui()
        if self.monitor is not None:
            self._update_report()
        self._refresh_hud(self.sim.t, self.sim.dt, alive, 0, 0.0)

    # ---------------------------------------------------------------- buttons / info panel (arcade.gui)
    def _build_ui(self):
        self.UI = arcade.gui.UIManager()
        self.UI.enable()
        root = arcade.gui.UIAnchorLayout()

        def button(text, width, fn):
            b = arcade.gui.UIFlatButton(text=text, width=width, height=30)
            b.event("on_click")(lambda event, fn=fn: fn())
            return b

        self.btn_labels = button("Labels: OFF", 130, self.cycle_labels)
        self.btn_live = button("Live print: OFF", 150, self.toggle_live_print)
        self.btn_pause = button("Pause", 90, self.toggle_pause)
        row = arcade.gui.UIBoxLayout(vertical=False, space_between=6)
        for b in (self.btn_labels, self.btn_live, button("Snapshot", 100, self.print_snapshot),
                  button("Stability", 100, self.print_stability), button("Speed -", 90, lambda: self.change_speed(1 / 1.5)),
                  button("Speed +", 90, lambda: self.change_speed(1.5)), self.btn_pause):
            row.add(b)
        root.add(row, anchor_x="left", anchor_y="top", align_x=10, align_y=-10)

        # top-right panel: simulated time, how long to keep running, who is alive
        self.ui_time = arcade.gui.UILabel("", text_color=arcade.color.WHITE, font_size=16)
        self.ui_stop = arcade.gui.UILabel("", text_color=arcade.color.LIGHT_GRAY, font_size=16)
        self.ui_alive = arcade.gui.UILabel("", text_color=arcade.color.LIGHT_GRAY, font_size=14)
        panel = arcade.gui.UIBoxLayout(vertical=True, space_between=2, align="right")
        for w in (self.ui_time, self.ui_stop, self.ui_alive):
            panel.add(w)
        root.add(panel, anchor_x="right", anchor_y="top", align_x=-12, align_y=-10)
        self.UI.add(root)

    def _nm(self, i):
        """Display label of body i: its number in ID mode, its name otherwise."""
        return str(i) if self.label_mode == "id" else self.names[i]

    def _tn(self, i):
        return self.names[int(i)]

    @staticmethod
    def _set_color(label, color):
        try:
            label.font_color = color
        except Exception:
            pass

    def _refresh_hud(self, t, dt, n_alive, steps, elapsed):
        if self.hud is not None and self.show_hud:
            self.hud.text = (f"speed {self.achieved_speed:.3f}/{self.sim_speed:.3f} yr/s  steps/frame={steps}  "
                             f"dt={dt:.1e}  work={elapsed * 1e3:.1f}ms")
        self.ui_time.text = f"t = {t:,.3f} yr    alive {n_alive}/{self.n0}"
        rep = self.report
        if self.monitor is None:
            text, col = "run time estimate: off (use --stability-every)", arcade.color.GRAY
        elif rep is None:
            text, col = "run time estimate: collecting data...", arcade.color.GRAY
        else:
            v, t_stop, rem = rep.stop_estimate()
            if v == "SETTLED":
                text, col = f"SETTLED - safe to stop (since t = {t_stop:,.2f} yr)", arcade.color.LIGHT_GREEN
            elif v == "PENDING":
                text, col = f"run until t = {t_stop:,.2f} yr  (+{rem:,.2f} yr)", arcade.color.YELLOW
            else:
                text, col = f"still evolving - earliest possible stop t = {t_stop:,.2f} yr", arcade.color.ORANGE
        self.ui_stop.text = text
        self._set_color(self.ui_stop, col)
        if n_alive <= 8 and self._last_buf is not None:
            buf = self._last_buf
            ids = sorted(int(i) for i in buf[buf[:, 2] > 0, 3])
            self.ui_alive.text = "alive: " + ", ".join(self._nm(i) for i in ids)
        else:
            self.ui_alive.text = ""

    # ---------------------------------------------------------------- actions (buttons and keys share these)
    def cycle_labels(self):
        self.label_mode = {"off": "id", "id": "name", "name": "off"}[self.label_mode]
        self.btn_labels.text = "Labels: " + self.label_mode.upper()
        self._labels.clear()

    def toggle_live_print(self):
        self.live_print = not self.live_print
        self.btn_live.text = "Live print: " + ("ON" if self.live_print else "OFF")
        self._last_print = 0.0

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.text = "Resume" if self.paused else "Pause"

    def change_speed(self, factor):
        self.sim_speed *= factor

    def print_snapshot(self):
        """One-shot table of the heaviest live bodies (positions in AU about the system centre of mass)."""
        d = self.sim.diagnostics()
        rows = self.sim.fetch_log(20)
        rows = rows[np.argsort(-rows[:, 4])] if len(rows) else rows
        print(f"--- t = {d['t']:.4f} yr | alive {d['alive']}/{self.n0} | E = {d['total']:.6e} | "
              f"Lz = {d['angular_momentum_z']:.5e} ---")
        print(f"{'id':>5} {'name':<12} {'mass':>9} {'x [AU]':>11} {'y [AU]':>11} {'vx':>11} {'vy':>11}   members")
        for x, y, vx, vy, m, i in rows:
            i = int(i)
            absorbed = len(self.members.get(i, [i])) - 1
            print(f"{i:>5} {self.names[i]:<12} {m:>9.4g} {x:>11.4e} {y:>11.4e} {vx:>11.4e} {vy:>11.4e}   "
                  f"{'+%d absorbed' % absorbed if absorbed else ''}")
        if d["alive"] > len(rows):
            print(f"      ... {d['alive'] - len(rows)} lighter bodies not shown")

    def _update_report(self):
        """Record a stability snapshot and turn the history into the current verdict / stop estimate."""
        try:
            if self.monitor is not None:
                self.report = _finalize_report(self.monitor.record(self.sim), self.monitor.params, self.monitor)
                return self.report
            return self.sim.stability_report(verbose=False)
        except Exception as e:  # never let diagnostics kill the window
            print("[stability] report failed:", e)
            return None

    def print_stability(self):
        rep = self._update_report()
        if rep is not None:
            print("[stability] " + (rep.summary() if VERBOSE else rep.brief(self._tn)))

    def _report_merges(self, n_alive):
        """Print every merge as one line and keep the 'who absorbed whom' bookkeeping for the labels."""
        events, self.sim.merge_log = self.sim.merge_log, []
        print("\n".join(merge_lines(events, self.names, self.members, n_alive)))

    def _live_print(self, t, dt, n_alive):
        now = time.perf_counter()
        if now - self._last_print < 1.0:
            return
        self._last_print = now
        print(f"t = {t:.3f} yr | alive {n_alive} | dt = {dt:.1e} | E = {float(self.sim.energy()):.6e} | "
              f"{self.achieved_speed:.2f} yr/s")

    def _init_point_renderer(self, n):
        try:
            from arcade.gl import BufferDescription
            self.pt_prog = self.ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
            self.pt_buf = self.ctx.buffer(reserve=n * 12, usage="dynamic")
            self.pt_geom = self.ctx.geometry(
                [BufferDescription(self.pt_buf, "2f 1f", ["in_pos", "in_radius"], instanced=True)],
                mode=self.ctx.TRIANGLE_STRIP)
            self.pt_ok = True
            say("[render]  GPU point renderer OK")
        except Exception as e:
            print("[render]  point shader failed, falling back to arcade.draw_points (slower):", e)
        self._draw_buf = np.empty((0, 4), np.float32)

    def _init_trails(self, buf):
        """Ring buffer indexed by ORIGINAL body id so compaction never invalidates a trail."""
        self.trails_gl = False
        self.cpu_trails = {}
        self.L = int(min(600, max(16, self.trail_budget // self.n0)))
        if self.trail_budget <= 0:
            return
        try:
            self.ctx.enable(self.ctx.BLEND)
            self.tr_prog = self.ctx.program(vertex_shader=TRAIL_VS, fragment_shader=TRAIL_FS)
            self.tr_buf = self.ctx.buffer(reserve=self.L * self.n0 * 8, usage="dynamic")
            self.tr_geom = self.ctx.geometry(mode=self.ctx.LINES)
            xy0 = np.zeros((self.n0, 2), np.float32)
            xy0[buf[:, 3].astype(np.int64)] = buf[:, :2]
            self.tr_last = xy0
            self.tr_buf.write(np.tile(xy0, (self.L, 1)).tobytes())      # no garbage before the ring fills
            self.tr_head = 0
            self.trails_gl = True
            say(f"[render]  trails: every body, {self.L} points each ({self.L * self.n0:,} total)")
        except Exception as e:
            print("[render]  GPU trail renderer failed:", e)
            if self.n0 <= self.sprite_mode_max:
                print("[render]  using the simple CPU trails (small N only)")
            else:
                print("[render]  trails disabled for large N (tell me this message and I will adapt the shader)")

    # ---------------------------------------------------------------- frame
    def on_update(self, delta_time):
        if self.paused:
            return
        t_start = time.perf_counter()
        budget = self.step_budget_ms / 1000

        # constant simulated time per frame; cap delta so a hiccup does not cause a burst of steps
        self.t_goal += self.sim_speed * min(delta_time, 1 / 30)
        t_before = self.sim.t
        steps = 0
        while self.sim.t < self.t_goal:
            self.sim.step()
            steps += 1
            if time.perf_counter() - t_start > budget:
                break
        if self.sim.t < self.t_goal:
            self.t_goal = self.sim.t          # cannot keep up -> slow motion, never accumulate debt
        self.last_steps = steps
        if delta_time > 0:
            self.achieved_speed = 0.9 * self.achieved_speed + 0.1 * (self.sim.t - t_before) / delta_time

        buf, t, dt, n_alive = self.sim.fetch_render(global_center_x, global_center_y)
        self._last_buf = buf
        elapsed = time.perf_counter() - t_start

        self._show(buf)
        self.frame_count += 1

        if self.sim.merge_log:
            self._report_merges(n_alive)

        if self.frame_count % self.trail_every == 0:
            self._push_trail(buf)

        if self.log_every and self.frame_count % self.log_every == 0:
            self._log(t)

        if self.group_every and self.frame_count % self.group_every == 0:
            self._log_groups(t)

        if self.monitor is not None and self.frame_count % self.stability_every == 0:
            self._update_report()

        if self.frame_count % 6 == 0:
            self._refresh_hud(t, dt, n_alive, steps, elapsed)

        if self.live_print:
            self._live_print(t, dt, n_alive)

        n_rows = buf.shape[0]
        if n_rows - n_alive >= max(16, self.dead_fraction * n_rows):
            self.sim.compact()

    def _push_trail(self, buf):
        alive = buf[:, 2] > 0
        if self.trails_gl:
            cur = self.tr_last.copy()                 # dead bodies keep repeating their last point
            cur[buf[alive, 3].astype(np.int64)] = buf[alive, :2]
            self.tr_buf.write(cur.tobytes(), offset=self.tr_head * self.n0 * 8)
            self.tr_head = (self.tr_head + 1) % self.L
            self.tr_last = cur
        elif self.n0 <= self.sprite_mode_max:
            for x, y, r, i in buf[alive]:
                self.cpu_trails.setdefault(int(i), deque(maxlen=600)).append((float(x), float(y)))

    def _show(self, buf):
        if self.use_sprites:
            for x, y, r, i in buf:
                bid = int(i)
                s = self.sprite_of.get(bid)
                if s is None:
                    continue
                if r <= 0:
                    self.planets.remove(s)
                    del self.sprite_of[bid]
                    continue
                s.center_x, s.center_y = float(x), float(y)
                if abs(r - s.r_px) > 1e-6:
                    s.update_texture(float(r), s.color)
        elif self.pt_ok:
            data = np.ascontiguousarray(buf[:, :3])
            self.pt_buf.write(data.tobytes())
            self._n_draw = data.shape[0]
        else:
            self._draw_buf = buf

    def _log(self, t):
        energy = float(self.sim.energy())
        drift = (energy - self.E0) / abs(self.E0) if self.E0 != 0 else 0.0
        lz = float(self.sim.angular_momentum().sum())
        self.energy_rows.append((t, energy, drift, lz))
        rows = self.sim.fetch_log(self.log_bodies)
        if rows.size:
            table = np.column_stack([np.full(len(rows), t), rows[:, 5], rows[:, 0:4], rows[:, 4]])
            np.savetxt(self.log_f, table, delimiter=",", fmt="%.9g")

    def _log_groups(self, t):
        groups = self.sim.bound_groups(min_size=2, tidal=False)
        self.group_rows.append((t, len(groups), groups[0].size if groups else 0,
                                sum(g.size for g in groups)))

    def on_draw(self):
        self.clear()
        self.camera.use()
        if self.trails_gl:
            self.tr_buf.bind_to_storage_buffer(binding=0)
            self.tr_prog["n_bodies"] = self.n0
            self.tr_prog["n_slots"] = self.L
            self.tr_prog["head"] = self.tr_head
            self.tr_prog["fade"] = 1.0 if self.trail_fade else 0.0
            self.tr_geom.render(self.tr_prog, mode=self.ctx.LINES, vertices=2 * (self.L - 1) * self.n0)
        else:
            for k, trail in self.cpu_trails.items():
                if len(trail) > 1:
                    arcade.draw_line_strip(list(trail), TRAIL_COLORS[k % len(TRAIL_COLORS)])
        if self.use_sprites:
            self.planets.draw()  # type: ignore
        elif self.pt_ok:
            self.pt_prog["zoom"] = float(self.camera.zoom)
            self.pt_geom.render(self.pt_prog, mode=self.ctx.TRIANGLE_STRIP, vertices=4, instances=self._n_draw)
        else:
            b = self._draw_buf
            arcade.draw_points([tuple(p) for p in b[b[:, 2] > 0][:20000, :2]], arcade.color.WHITE, 2)
        # screen-space layer: labels, HUD, buttons
        self.default_camera.use()
        self._draw_labels()
        if self.hud is not None and self.show_hud:
            try:
                self.hud.draw()
            except Exception:
                self.hud = None
        self.UI.draw()

    def _draw_labels(self):
        """Name / number next to every live body. World -> screen by hand: the label stays the same size at any zoom."""
        buf = self._last_buf
        if self.label_mode == "off" or buf is None:
            return
        rows = buf[buf[:, 2] > 0]
        if len(rows) > self.label_max:
            rows = rows[np.argsort(-rows[:, 2])[:self.label_max]]      # biggest bodies first
        cx, cy = self.camera.position[0], self.camera.position[1]
        z, hw, hh = float(self.camera.zoom), self.width / 2, self.height / 2
        live = set()
        for x, y, r, i in rows:
            bid = int(i)
            live.add(bid)
            sx, sy = (float(x) - cx) * z + hw, (float(y) - cy) * z + hh
            if sx < -80 or sy < -20 or sx > self.width + 80 or sy > self.height + 20:
                continue
            txt = _tracker_label(self.label_mode, bid, self.names, self.members)
            lab = self._labels.get(bid)
            if lab is None:
                lab = self._labels[bid] = arcade.Text(txt, 0, 0, arcade.color.WHITE, 12)
            elif lab.text != txt:
                lab.text = txt
            off = max(float(r) * z, 2.0) + 4.0
            lab.x, lab.y = sx + off, sy + off
            lab.draw()
        for bid in [b for b in self._labels if b not in live]:         # merged away -> free the label
            del self._labels[bid]

    def on_close(self):
        try:
            self.log_f.close()
            if self.energy_rows:
                np.savetxt("simulation_energy.csv", np.array(self.energy_rows), delimiter=",",
                           header="time,total_energy,relative_energy_drift,angular_momentum_z", comments="")
            if self.group_rows:
                np.savetxt("simulation_groups.csv", np.array(self.group_rows), delimiter=",", fmt="%.9g",
                           header="time,n_groups,largest_group,bodies_in_groups", comments="")
        except Exception as e:
            print("Error saving data:", e)
        try:
            if self.sim.mass.shape[0] <= 2000:
                rep = self._update_report()
                if rep is not None:
                    print("\nFinal state  " + (rep.summary() if VERBOSE else rep.brief(self._tn)))
        except Exception as e:
            print("stability report failed:", e)
        super().on_close()

    # ---------------------------------------------------------------- input
    def on_key_press(self, symbol, modifiers):
        global global_center_x, global_center_y
        if symbol == arcade.key.F:
            self.is_fullscreen = not self.is_fullscreen
            self.set_fullscreen(self.is_fullscreen)
            if not self.is_fullscreen:
                self.set_size(self.window_w, self.window_h)
                self.set_location(0, 38)
            global_center_x = self.center_x
            global_center_y = self.center_y
        elif symbol == arcade.key.SPACE:
            self.toggle_pause()
        elif symbol == arcade.key.P:
            self.toggle_live_print()
        elif symbol == arcade.key.L:
            self.cycle_labels()
        elif symbol == arcade.key.N:
            self.print_snapshot()
        elif symbol == arcade.key.E:
            self.sim.set_fixed_dt(not self.sim.fixed_dt)
            print(f"Fixed-dt mode: {'ON' if self.sim.fixed_dt else 'OFF'} (dt = {self.sim.fixed_dt_value})")
        elif symbol == arcade.key.UP:
            self.change_speed(1.5)
        elif symbol == arcade.key.DOWN:
            self.change_speed(1 / 1.5)
        elif symbol == arcade.key.RIGHT:
            self.step_budget_ms = min(14.0, self.step_budget_ms + 1.0)
        elif symbol == arcade.key.LEFT:
            self.step_budget_ms = max(1.0, self.step_budget_ms - 1.0)
        elif symbol == arcade.key.A:
            self.show_hud = not self.show_hud
        elif symbol == arcade.key.T:
            self.trail_fade = not self.trail_fade
        elif symbol == arcade.key.B:
            self.sim.calculate_dynamics()
        elif symbol == arcade.key.S:
            self.print_stability()
        elif symbol == arcade.key.D:
            d = self.sim.diagnostics()
            print(f"t={d['t']:.4f} yr  alive={d['alive']}  KE={d['kinetic']:.6e}  PE={d['potential']:.6e}  "
                  f"E={d['total']:.6e}  virial 2KE/|PE|={d['virial']:.3f}  Lz={d['angular_momentum_z']:.6e}")

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # the buttons consume their own clicks/drags (UIManager handles events first), so this only pans the scene
        if buttons & arcade.MOUSE_BUTTON_LEFT:
            z = self.camera.zoom
            self.camera.position = (self.camera.position[0] - dx / z, self.camera.position[1] - dy / z)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            self.camera.zoom *= 1.1
        elif scroll_y < 0:
            self.camera.zoom /= 1.1


if __name__ == "__main__":
    if "--headless" in sys.argv:               # no window: hand over to the fast runner (all its options apply)
        import nbody_headless
        sys.exit(nbody_headless.main([a for a in sys.argv[1:] if a != "--headless"]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print the set-up details and the long reports (default: only events and short reports)")
    ap.add_argument("--headless", action="store_true",
                    help="no window: run nbody_headless.py (much faster; see python nbody_headless.py -h)")
    ap.add_argument("--float32", action="store_true", help="everything in float32")
    ap.add_argument("--force-fp32", action="store_true", help="(now the default on CUDA) forces in float32")
    ap.add_argument("--fp64-force", action="store_true", help="forces in float64 (slow on consumer GPUs)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the EAGER force tile (needs Triton)")
    ap.add_argument("--kernel", choices=["auto", "eager", "cupy"], default="auto",
                    help="force backend: auto = fused CuPy kernels if they work, else PyTorch")
    ap.add_argument("--random", type=int, default=0, help="stress scene with N bodies")
    ap.add_argument("--bench", action="store_true", help="headless component timing table")
    ap.add_argument("--check", action="store_true",
                    help="headless check of the body-state diagnostics against the NumPy formulas")
    ap.add_argument("--trail-budget", type=int, default=3_000_000,
                    help="total trail points over ALL bodies (per-body length = budget/N, 16..600); 0 = off")
    ap.add_argument("--log-bodies", type=int, default=256, help="log the K heaviest bodies (0 = all)")
    ap.add_argument("--log-every", type=int, default=10, help="log every F frames (0 = off)")
    ap.add_argument("--group-every", type=int, default=0,
                    help="every F frames, log bound-group statistics to simulation_groups.csv (0 = off; O(N^2) per log)")
    ap.add_argument("--stability-every", type=int, default=-1,
                    help="record a stability snapshot every F frames for the long-term verdict (S key / on exit); "
                         "-1 = auto (30 if N <= 1000, else off), 0 = off")
    ap.add_argument("--trail-fade", action="store_true",
                    help="fade older trail points out (default: off, trails keep full opacity)")
    ap.add_argument("--set-high-perf-gpu", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    if args.set_high_perf_gpu:
        set_windows_gpu_preference()
        sys.exit(0)

    device = pick_device()
    dtype = torch.float32 if args.float32 else torch.float64
    use_fp32_force = args.float32 or args.force_fp32 or (device.type == "cuda" and not args.fp64_force)
    fdtype = torch.float32 if use_fp32_force else None

    if args.check:
        sys.exit(0 if run_analysis_check(device) else 1)

    if args.bench:
        run_bench(device, dtype, fdtype, args.compile, args.kernel)
        sys.exit(0)

    RANDOM_N = args.random
    Env = myEnvironment(title="N-body (GPU v5)", device=device, dtype=dtype, force_dtype=fdtype,
                        use_compile=args.compile, kernel=args.kernel, trail_budget=args.trail_budget,
                        log_bodies=(args.log_bodies or None), log_every=args.log_every,
                        trail_fade=args.trail_fade, group_every=args.group_every,
                        stability_every=args.stability_every)
    Env.setup()
    arcade.run()