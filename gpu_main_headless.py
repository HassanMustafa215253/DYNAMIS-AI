"""
Headless (no window) runner for the N-body simulation.

    python nbody_headless.py                      # the scene in gpu_main.build_scene(), run until it settles
    python nbody_headless.py --random 500 --t-max 2000
    python gpu_main.py --headless ...             # same thing through the GUI script

What it does
  * Runs the SAME physics as the window (same adaptive dt rule, same softening, same merge rule, same stability
    analysis from gpu_main.py) but with nothing in the way: no drawing, no frame limit, no per-frame GPU->CPU
    copies, no real-time pacing. The window advances 0.5 simulated years per real second by default; this runs as
    fast as the machine allows.
  * Backends
      cpu  a numba-compiled integrator in float64 (the default for N <= 1500). For small systems the GPU is the
           slow option: every step needs ~10 tiny kernel launches plus a GPU->CPU sync to pick the next dt, so a
           3-body run spends almost all its time waiting on latency. The compiled CPU loop does a whole step in
           microseconds and runs thousands of steps per call.
      gpu  GPUNBody on CUDA, exactly the engine the window uses (best for large N).
  * Stops by itself: as soon as the stability monitor says SETTLED (every body is in a stable bound structure or
    has escaped, and nothing changed for the required window), or at --t-max / --wall-max. Ctrl-C also ends the run
    cleanly and still writes the result.
  * Output: a few progress lines, one line per merge (while there are few), a short final report, and a JSON file
    (--out) with the verdict, the time the outcome was reached, the fate of every body and the merge history.
    That JSON is the thing to collect when comparing runs.

Accuracy notes
  * The cpu backend integrates in float64. The window on CUDA computes forces in float32 by default. For a chaotic
    system (a 3-body scatter is one) the two will diverge after a while even though both are correct; compare
    statistics over many runs, not single trajectories. Use --backend gpu to reproduce the window exactly.
  * --dt-div is the one speed/accuracy knob: dt = (adaptive step) / dt_div. 3 is the window's value. Raising the
    step (smaller dt_div) is faster; always check the energy drift line of the report afterwards.

Needs: numpy, torch (the stability analysis runs on torch), arcade (gpu_main imports it), numba (cpu backend).
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

import gpu_main as gm
from gpu_main import AU_p_Pix, EPS2, G, GPUNBody, StabilityMonitor, StabilityParams, _finalize_report

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                                  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        return a[0] if (len(a) == 1 and callable(a[0])) else (lambda f: f)
    prange = range


# ====================================================================================== compiled kernels (cpu backend)
@njit(cache=True)
def _accel_serial(n, px, py, m, ax, ay, phi, g, eps2):
    """Softened accelerations, potential sums phi_i = sum_j m_j / r_ij and the closest pair, each pair visited once."""
    for i in range(n):
        ax[i] = 0.0
        ay[i] = 0.0
        phi[i] = 0.0
    best = np.inf
    bi = 0
    bj = 0
    for i in range(n):
        xi = px[i]
        yi = py[i]
        mi = m[i]
        for j in range(i + 1, n):
            dx = px[j] - xi
            dy = py[j] - yi
            r2 = dx * dx + dy * dy
            inv = 1.0 / np.sqrt(r2 + eps2)
            inv3 = inv * inv * inv
            wj = m[j] * inv3
            wi = mi * inv3
            ax[i] += dx * wj
            ay[i] += dy * wj
            ax[j] -= dx * wi
            ay[j] -= dy * wi
            phi[i] += m[j] * inv
            phi[j] += mi * inv
            if r2 < best:
                best = r2
                bi = i
                bj = j
    for i in range(n):
        ax[i] *= g
        ay[i] *= g
    return best, bi, bj


@njit(cache=True, parallel=True)
def _accel_parallel(n, px, py, m, ax, ay, phi, g, eps2, bestr, bestj):
    """Same result, rows split over threads (twice the flops, no write conflicts). Pays off for larger n."""
    for i in prange(n):
        xi = px[i]
        yi = py[i]
        sx = 0.0
        sy = 0.0
        sp = 0.0
        b = np.inf
        jb = 0
        for j in range(n):
            if j == i:
                continue
            dx = px[j] - xi
            dy = py[j] - yi
            r2 = dx * dx + dy * dy
            inv = 1.0 / np.sqrt(r2 + eps2)
            w = m[j] * inv * inv * inv
            sx += dx * w
            sy += dy * w
            sp += m[j] * inv
            if r2 < b:
                b = r2
                jb = j
        ax[i] = g * sx
        ay[i] = g * sy
        phi[i] = sp
        bestr[i] = b
        bestj[i] = jb
    best = np.inf
    bi = 0
    for i in range(n):
        if bestr[i] < best:
            best = bestr[i]
            bi = i
    return best, bi, bestj[bi]


@njit(cache=True)
def _compute_dt(r2, v2, a2, rv2, ra2, dt_prev, pp):
    """Line-for-line port of gpu_main.compute_dt. pp = [dt_min, dt_max, close_thr, close_safety, close_min_dt, dt_div]."""
    inf = np.inf
    min_dist = max(np.sqrt(r2), 1e-12) if np.isfinite(r2) else inf
    vmax = np.sqrt(v2)
    amax = np.sqrt(a2)
    dt_vel = 0.2 * min_dist / vmax if vmax > 0 else inf
    dt_acc = 0.2 * np.sqrt(min_dist / amax) if amax > 0 else inf
    c = min(pp[1], min(dt_vel, dt_acc))
    if min_dist < pp[2]:
        rel_v = np.sqrt(rv2)
        rel_a = np.sqrt(ra2)
        cc = inf
        if rel_v > 0:
            cc = pp[3] * min_dist / rel_v
        if rel_a > 0:
            cc = min(cc, pp[3] * np.sqrt(min_dist / rel_a))
        c = min(c, max(pp[4], cc) if cc < inf else pp[4])
    target = max(pp[0], c) / pp[5]
    return min(target, dt_prev * 1.2)


@njit(cache=True)
def _merge(n, px, py, vx, vy, m, rad, ids, contact, ev_lose, ev_win, ev_mass):
    """Port of GPUNBody._merge: every overlapping cluster becomes its heaviest member (mass, momentum conserved,
    radius from summed volume). Losers are removed and the arrays compacted. Returns (new n, number of losers)."""
    order = np.argsort(m[:n], kind="mergesort")            # stable ascending mass; ties -> higher index heavier
    rank = np.empty(n, np.int64)
    for k in range(n):
        rank[order[k]] = k
    target = np.empty(n, np.int64)
    moved = False
    for i in range(n):
        b = rank[i]
        for j in range(n):
            if j != i:
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                s = (rad[i] + rad[j]) * contact
                if dx * dx + dy * dy < s * s and rank[j] > b:
                    b = rank[j]
        target[i] = order[b]
        if target[i] != i:
            moved = True
    if not moved:
        return n, 0
    for _ in range(3):                                      # pointer jumping
        t2 = np.empty(n, np.int64)
        for i in range(n):
            t2[i] = target[target[i]]
        target = t2
    t2 = np.empty(n, np.int64)
    for i in range(n):
        t2[i] = target[target[i]]
    for i in range(n):
        if t2[i] != target[i]:                              # not converged -> stays its own root
            target[i] = i
    cnt = np.zeros(n, np.int64)
    for i in range(n):
        cnt[target[i]] += 1
    M = np.zeros(n)
    Px = np.zeros(n)
    Py = np.zeros(n)
    Vx = np.zeros(n)
    Vy = np.zeros(n)
    R3 = np.zeros(n)
    for i in range(n):
        t = target[i]
        if cnt[t] > 1:
            M[t] += m[i]
            Px[t] += m[i] * px[i]
            Py[t] += m[i] * py[i]
            Vx[t] += m[i] * vx[i]
            Vy[t] += m[i] * vy[i]
            R3[t] += rad[i] ** 3
    keep = np.ones(n, np.bool_)
    nev = 0
    for i in range(n):
        t = target[i]
        if cnt[t] > 1:
            if t == i:
                px[i] = Px[i] / M[i]
                py[i] = Py[i] / M[i]
                vx[i] = Vx[i] / M[i]
                vy[i] = Vy[i] / M[i]
                m[i] = M[i]
                rad[i] = R3[i] ** (1.0 / 3.0)
            else:
                ev_lose[nev] = ids[i]
                ev_win[nev] = ids[t]
                ev_mass[nev] = M[t]
                nev += 1
                keep[i] = False
    k = 0
    for i in range(n):
        if keep[i]:
            if k != i:
                px[k] = px[i]
                py[k] = py[i]
                vx[k] = vx[i]
                vy[k] = vy[i]
                m[k] = m[i]
                rad[k] = rad[i]
                ids[k] = ids[i]
            k += 1
    return k, nev


@njit(cache=True)
def _stats_and_dt(n, vx, vy, ax, ay, best, bi, bj, dt, pp):
    v2 = 0.0
    a2 = 0.0
    for i in range(n):
        v2 = max(v2, vx[i] * vx[i] + vy[i] * vy[i])
        a2 = max(a2, ax[i] * ax[i] + ay[i] * ay[i])
    rv2 = 0.0
    ra2 = 0.0
    if n > 1:
        dvx = vx[bi] - vx[bj]
        dvy = vy[bi] - vy[bj]
        dax = ax[bi] - ax[bj]
        day = ay[bi] - ay[bj]
        rv2 = dvx * dvx + dvy * dvy
        ra2 = dax * dax + day * day
    return _compute_dt(best, v2, a2, rv2, ra2, dt, pp)


@njit(cache=True)
def _accel(n, px, py, m, ax, ay, phi, g, eps2, par, bestr, bestj):
    if par:
        return _accel_parallel(n, px, py, m, ax, ay, phi, g, eps2, bestr, bestj)
    return _accel_serial(n, px, py, m, ax, ay, phi, g, eps2)


@njit(cache=True)
def _advance(px, py, vx, vy, m, rad, ids, ax, ay, phi, sc, ic, pp, mp, bestr, bestj,
             ev_lose, ev_win, ev_mass, t_end, max_steps):
    """Kick-drift-merge-force-kick exactly as GPUNBody.step(), repeated until t >= t_end, max_steps steps, or a
    merge happened (code 2; the caller reads the events and calls again).
      sc = [t, dt, closest r^2 (previous step), rmax_px]      ic = [n, closest i, closest j, n_events, par_n]
      mp = [G, eps2, merge reach factor, contact factor]"""
    n = ic[0]
    steps = 0
    code = 0
    while sc[0] < t_end and steps < max_steps:
        dt = sc[1]
        h = 0.5 * dt
        for i in range(n):
            vx[i] += h * ax[i]
            vy[i] += h * ay[i]
        for i in range(n):
            px[i] += dt * vx[i]
            py[i] += dt * vy[i]
        merged = False
        reach = mp[2] * sc[3]
        if sc[2] < reach * reach:
            n2, nev = _merge(n, px, py, vx, vy, m, rad, ids, mp[3], ev_lose, ev_win, ev_mass)
            if nev > 0:
                n = n2
                ic[0] = n
                ic[3] = nev
                merged = True
                rm = 0.0
                for i in range(n):
                    rm = max(rm, rad[i])
                sc[3] = rm
        best, bi, bj = _accel(n, px, py, m, ax, ay, phi, mp[0], mp[1], n >= ic[4], bestr, bestj)
        for i in range(n):
            vx[i] += h * ax[i]
            vy[i] += h * ay[i]
        sc[0] += dt
        ic[1] = bi
        ic[2] = bj
        sc[2] = best
        sc[1] = _stats_and_dt(n, vx, vy, ax, ay, best, bi, bj, dt, pp)
        steps += 1
        if merged:
            code = 2
            break
    return steps, code


# ====================================================================================== engines
class FastNBody:
    """numba float64 engine with the interface the runner needs. State lives in numpy arrays; dead bodies are
    removed at once (no zero-mass rows), and `ids` keeps the original body numbers."""

    def __init__(self, pos, vel, mass, radius_px, dtp, collision_scale=1.0, merge_safety=1.5, parallel_n=600):
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed (pip install numba) -> use --backend gpu")
        n = len(mass)
        self.n0 = n
        self.px = np.ascontiguousarray(pos[:, 0], dtype=np.float64)
        self.py = np.ascontiguousarray(pos[:, 1], dtype=np.float64)
        self.vx = np.ascontiguousarray(vel[:, 0], dtype=np.float64)
        self.vy = np.ascontiguousarray(vel[:, 1], dtype=np.float64)
        self.m = np.array(mass, dtype=np.float64)
        self.rad = np.array(radius_px, dtype=np.float64)
        self.ids = np.arange(n, dtype=np.int64)
        self.ax, self.ay, self.phi = np.zeros(n), np.zeros(n), np.zeros(n)
        self.bestr, self.bestj = np.zeros(n), np.zeros(n, np.int64)
        self.ev_lose, self.ev_win = np.zeros(n, np.int64), np.zeros(n, np.int64)
        self.ev_mass = np.zeros(n)
        self.pp = np.array([dtp["dt_min"], dtp["dt_max"], dtp["close_threshold"], dtp["close_safety"],
                            dtp["close_min_dt"], dtp.get("dt_div", 3.0)], dtype=np.float64)
        contact = AU_p_Pix * collision_scale
        self.mp = np.array([G, EPS2, merge_safety * 2.0 * contact, contact], dtype=np.float64)
        self.sc = np.zeros(4)
        self.ic = np.array([n, 0, 0, 0, parallel_n], dtype=np.int64)
        self.merge_log = []
        self.steps = 0
        best, bi, bj = _accel(n, self.px, self.py, self.m, self.ax, self.ay, self.phi, G, EPS2,
                              n >= parallel_n, self.bestr, self.bestj)
        self.ic[1], self.ic[2] = bi, bj
        self.sc[2] = best
        self.sc[3] = self.rad.max()
        self.sc[1] = _stats_and_dt(n, self.vx, self.vy, self.ax, self.ay, best, bi, bj, 0.01, self.pp)
        self._view = None

    # -- state
    @property
    def t(self):
        return float(self.sc[0])

    @property
    def dt(self):
        return float(self.sc[1])

    @property
    def n_alive(self):
        return int(self.ic[0])

    def energy(self):
        n = self.n_alive
        kin = 0.5 * float((self.m[:n] * (self.vx[:n] ** 2 + self.vy[:n] ** 2)).sum())
        return kin - 0.5 * G * float((self.m[:n] * self.phi[:n]).sum())

    def alive_ids(self):
        return sorted(int(i) for i in self.ids[:self.n_alive])

    def rows(self, k=256):
        """x y vx vy m id of the k heaviest live bodies."""
        n = self.n_alive
        idx = np.argsort(-self.m[:n], kind="stable")[:k]
        return np.column_stack([self.px[idx], self.py[idx], self.vx[idx], self.vy[idx], self.m[idx], self.ids[idx]])

    # -- run
    def advance(self, t_end, deadline=None, chunk=2_000_000):
        """Integrate until t >= t_end (or the wall-clock `deadline`). Merge events go to self.merge_log."""
        while self.sc[0] < t_end and self.n_alive > 1:
            steps, code = _advance(self.px, self.py, self.vx, self.vy, self.m, self.rad, self.ids, self.ax, self.ay,
                                   self.phi, self.sc, self.ic, self.pp, self.mp, self.bestr, self.bestj,
                                   self.ev_lose, self.ev_win, self.ev_mass, t_end, chunk)
            self.steps += steps
            if code == 2:
                k = int(self.ic[3])
                self.merge_log.append((self.t, self.ev_lose[:k].tolist(), self.ev_win[:k].tolist(),
                                       self.ev_mass[:k].tolist()))
            if deadline is not None and time.perf_counter() > deadline:
                break

    # -- analysis: the stability code of gpu_main works on a GPUNBody, so hand it one filled with our state
    def view(self):
        n = self.n_alive
        v = self._view
        if v is None:
            v = self._view = GPUNBody(np.stack([self.px, self.py], 1), np.stack([self.vx, self.vy], 1), self.m,
                                      self.rad, torch.device("cpu"), torch.float64, kernel="eager")
            v.track_merges = False
        v.pos = torch.from_numpy(np.stack([self.px[:n], self.py[:n]], 1))
        v.vel = torch.from_numpy(np.stack([self.vx[:n], self.vy[:n]], 1))
        v.mass = torch.from_numpy(self.m[:n].copy())
        v.rad = torch.from_numpy(self.rad[:n].copy())
        v.ids = torch.from_numpy(self.ids[:n].copy())
        v.acc = torch.from_numpy(np.stack([self.ax[:n], self.ay[:n]], 1))
        v._phi = torch.from_numpy(self.phi[:n].copy())
        v.t = self.t
        return v


class GpuEngine:
    """The window's own engine (GPUNBody), driven without a window."""

    def __init__(self, pos, vel, mass, radius_px, dtp, kernel="auto", device=None):
        self.dev = device or gm.pick_device()
        fd = torch.float32 if self.dev.type == "cuda" else None
        self.sim = GPUNBody(pos, vel, mass, radius_px, self.dev, torch.float64, fd, kernel=kernel)
        self.sim.dtp.update({k: v for k, v in dtp.items() if k in self.sim.dtp})
        self.sim.dt = gm.compute_dt(self.sim._stats, 0.01, self.sim.dtp)
        self.n0 = len(mass)
        self.steps = 0

    t = property(lambda self: self.sim.t)
    dt = property(lambda self: self.sim.dt)
    merge_log = property(lambda self: self.sim.merge_log, lambda self, v: setattr(self.sim, "merge_log", v))

    @property
    def n_alive(self):
        return int((self.sim.mass > 0).sum())

    def energy(self):
        return float(self.sim.energy())

    def alive_ids(self):
        return sorted(int(i) for i in self.sim.ids[self.sim.mass > 0].cpu().tolist())

    def rows(self, k=256):
        r = self.sim.fetch_log(k)
        return r[np.argsort(-r[:, 4])] if len(r) else r

    def advance(self, t_end, deadline=None):
        sim, k = self.sim, 0
        while sim.t < t_end:
            sim.step()
            k += 1
            if (k & 127) == 0:
                if deadline is not None and time.perf_counter() > deadline:
                    break
                if sim.mass.shape[0] - int((sim.mass > 0).sum()) >= max(16, 0.05 * sim.mass.shape[0]):
                    sim.compact()
        self.steps += k
        if self.n_alive == 1:
            sim.t = max(sim.t, t_end)

    def view(self):
        return self.sim


# ====================================================================================== runner
def _fmt_wall(s):
    return f"{s:.1f} s" if s < 120 else (f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.2f} h")


def run(args):
    gm.VERBOSE = args.verbose
    pos, vel, mass, rad, _colors, names = gm.build_scene(random_n=args.random or None, seed=args.seed)
    n0 = len(mass)

    backend = args.backend
    if backend == "auto":
        backend = "cpu" if (HAVE_NUMBA and (n0 <= args.cpu_max_n or not torch.cuda.is_available())) else "gpu"
    dtp = dict(dt_min=args.dt_min, dt_max=20.0, close_threshold=0.75, close_safety=0.2, close_min_dt=1e-6,
               dt_div=args.dt_div)
    t_c = time.perf_counter()
    if backend == "cpu":
        eng = FastNBody(pos, vel, mass, rad, dtp)
        label = f"cpu (numba, float64, {'parallel' if n0 >= 600 else 'serial'})"
    else:
        eng = GpuEngine(pos, vel, mass, rad, dtp, kernel=args.kernel)
        label = f"gpu ({eng.sim.backend_name}, forces {'fp32' if eng.sim.force_dtype == torch.float32 else 'fp64'})"
    params = StabilityParams()
    if args.min_dyn is not None:
        params.min_dyn = args.min_dyn
    if args.min_periods is not None:
        params.min_periods = args.min_periods
    mon = StabilityMonitor(params, max_records=500_000)
    print(f"[headless] N={n0} | backend: {label} | t_max {args.t_max:g} yr | "
          f"{'runs to t_max' if args.no_stop else 'stops when SETTLED'}")

    def snapshot():
        t0 = time.perf_counter()
        r = _finalize_report(mon.record(eng.view()), params, mon)
        return r, time.perf_counter() - t0

    t_start = time.perf_counter()
    deadline = (t_start + args.wall_max) if args.wall_max else None
    rep, c_snap = snapshot()
    first_done = time.perf_counter()
    t_compile = first_done - t_c
    base = args.monitor_dt or max(0.5 * (mon.t_dyn0 or 1.0), 1e-6)
    t_mon = eng.t + base
    t_log = eng.t if args.log_dt else math.inf
    traj = None
    if args.log_dt:
        traj = open(args.trajectory_csv, "w")
        traj.write("time,object_id,x,y,vx,vy,mass\n")
    all_events, printed_merges = [], 0
    last_prog, last_t, last_wall = time.perf_counter(), eng.t, time.perf_counter()
    speed = 0.0
    E0 = eng.energy()
    stop_reason = None
    try:
        while True:
            # ---- verdict
            if rep.settled and not args.no_stop:
                stop_reason = "settled"
                break
            if eng.t >= args.t_max:
                stop_reason = "t_max"
                break
            if deadline and time.perf_counter() > deadline:
                stop_reason = "wall_max"
                break
            if eng.n_alive <= 1:                       # one body left: nothing can happen any more
                stop_reason = "settled"
                break
            # ---- advance to the next event (snapshot time, log time, t_max)
            t_target = min(t_mon, t_log, args.t_max)
            w0, s0 = time.perf_counter(), eng.t
            eng.advance(t_target, deadline)
            dw = time.perf_counter() - w0
            if dw > 1e-3 and eng.t > s0:
                inst = (eng.t - s0) / dw
                speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
            # ---- merge events
            ev, eng.merge_log = eng.merge_log, []
            if ev:
                all_events.extend(ev)
                if printed_merges < args.max_merge_lines:
                    for line in gm.merge_lines(ev, names, None, eng.n_alive):
                        print(line)
                        printed_merges += 1
                    if printed_merges >= args.max_merge_lines:
                        print(f"[headless] (further merges not printed individually; see the progress lines / {args.out})")
            # ---- structure snapshot
            take = eng.t >= t_mon or (ev and eng.n_alive <= 64) or eng.t >= args.t_max
            if take:
                rep, c_snap = snapshot()
                req = rep.required_span
                over = 4.0 * c_snap * speed                          # keep analysis to ~20% of the wall time ...
                # ... but never fewer than ~8 snapshots per required quiet window, ideally 20
                interval = min(max(req / 20.0, over), req / 8.0) if req > 0 else max(base, over)
                t_mon = eng.t + (args.monitor_dt or max(interval, 1e-9))
            if eng.t >= t_log:
                r = eng.rows(256)
                if len(r):
                    np.savetxt(traj, np.column_stack([np.full(len(r), eng.t), r[:, 5], r[:, 0:4], r[:, 4]]),
                               delimiter=",", fmt="%.9g")
                t_log = eng.t + args.log_dt
            # ---- progress
            now = time.perf_counter()
            if args.progress and now - last_prog >= args.progress:
                _v, t_stop, rem = rep.stop_estimate()
                est = {"SETTLED": "settled", "PENDING": f"run until t = {t_stop:,.1f} yr",
                       "NOT SETTLED": f"still evolving (earliest stop t = {t_stop:,.1f} yr)"}[rep.verdict]
                print(f"[t = {eng.t:12,.2f} yr] alive {eng.n_alive:>5} | {est} | {speed:,.0f} yr/s | "
                      f"wall {_fmt_wall(now - t_start)}")
                last_prog = now
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\n[headless] interrupted - writing the result so far")
    if traj:
        traj.close()

    # ---- final report
    wall = time.perf_counter() - t_start
    rep, _ = snapshot()
    verdict, t_stop, _rem = rep.stop_estimate()
    E = eng.energy()
    head = {"settled": f"SETTLED at t = {eng.t:,.3f} yr",
            "t_max": f"t_max reached at t = {eng.t:,.2f} yr, verdict {rep.verdict}",
            "wall_max": f"wall-clock limit reached at t = {eng.t:,.2f} yr, verdict {rep.verdict}",
            "interrupted": f"interrupted at t = {eng.t:,.2f} yr, verdict {rep.verdict}"}[stop_reason]
    print(f"\n[result] {head}   (wall {_fmt_wall(wall)}, {eng.steps:,} steps, "
          f"{eng.t / max(wall, 1e-9):,.0f} yr/s)")
    if rep.verdict == "SETTLED":
        print(f"         outcome was already settled by t = {t_stop:,.3f} yr -> a run of that length is enough "
              f"for this scene")
    elif rep.verdict == "PENDING":
        print(f"         not confirmed yet: keep running until t = {t_stop:,.2f} yr (+{rep.stop_estimate()[2]:,.2f} yr) "
              f"if nothing changes")
    else:
        print(f"         not settled: earliest possible stop t = {t_stop:,.2f} yr, only once the issues below clear")
    print("         " + rep.brief(lambda i: names[int(i)]).replace("\n", "\n         "))
    if gm.VERBOSE:
        print(rep.summary())

    result = dict(
        scene=("random N=%d seed=%d" % (n0, args.seed)) if args.random else "build_scene()",
        n_initial=n0, backend=label, stop_reason=stop_reason, verdict=rep.verdict,
        t_final=eng.t, settled_by_t=(t_stop if rep.verdict == "SETTLED" else None),
        required_window=rep.required_span, quiet_span=rep.quiet_span,
        alive=eng.alive_ids(), alive_names=[names[i] for i in eng.alive_ids()],
        fates={str(k): v for k, v in sorted(rep.fates.items())},
        groups=[dict(ids=g.ids, kind=g.kind, verdict=g.verdict, margin=(None if g.margin != g.margin else g.margin))
                for g in rep.groups],
        escaping=[dict(ids=list(k), distance_au=d["D"]) for k, d in rep.escaping.items()],
        energy_initial=E0, energy_final=E,
        energy_drift_since_last_merger=(None if rep.energy_drift != rep.energy_drift else rep.energy_drift),
        merges=[dict(t=t, absorbed=l, by=w, mass=m) for t, ls, ws, ms in all_events
                for l, w, m in zip(ls, ws, ms)],
        wall_seconds=wall, compile_and_setup_seconds=t_compile, steps=eng.steps,
        params=dict(dt_div=args.dt_div, dt_min=args.dt_min, min_dyn=params.min_dyn, min_periods=params.min_periods,
                    t_max=args.t_max, seed=args.seed))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[headless] wrote {args.out}" + (f" and {args.trajectory_csv}" if args.log_dt else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Headless N-body run: as fast as possible, stops when the outcome has settled.")
    ap.add_argument("--random", type=int, default=0, help="random stress scene with N bodies (default: build_scene())")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random scene")
    ap.add_argument("--t-max", type=float, default=1e5, help="stop at this simulated time [yr] if not settled")
    ap.add_argument("--wall-max", type=float, default=0, help="stop after this many real seconds (0 = no limit)")
    ap.add_argument("--no-stop", action="store_true", help="do not stop when SETTLED; run to --t-max")
    ap.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto",
                    help="auto = cpu (numba) for N <= --cpu-max-n, else gpu")
    ap.add_argument("--cpu-max-n", type=int, default=1500, help="largest N for which auto picks the cpu backend")
    ap.add_argument("--kernel", choices=["auto", "eager", "cupy"], default="auto", help="gpu backend force kernel")
    ap.add_argument("--dt-div", type=float, default=3.0, help="step divisor; 3 = same as the window, 1 = 3x bigger steps")
    ap.add_argument("--dt-min", type=float, default=1e-3, help="dt floor before the divisor (the window's dtp['dt_min'])")
    ap.add_argument("--min-dyn", type=float, default=None, help="stability window: dynamical times (default 10)")
    ap.add_argument("--min-periods", type=float, default=None, help="stability window: longest periods (default 50)")
    ap.add_argument("--monitor-dt", type=float, default=0, help="fixed snapshot interval [yr] (default: automatic)")
    ap.add_argument("--out", default="headless_result.json", help="result file")
    ap.add_argument("--log-dt", type=float, default=0, help="also log trajectories every this many yr (0 = off)")
    ap.add_argument("--trajectory-csv", default="headless_trajectory.csv")
    ap.add_argument("--progress", type=float, default=5.0, help="seconds between progress lines (0 = none)")
    ap.add_argument("--max-merge-lines", type=int, default=40, help="print at most this many merge lines")
    ap.add_argument("--verbose", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())