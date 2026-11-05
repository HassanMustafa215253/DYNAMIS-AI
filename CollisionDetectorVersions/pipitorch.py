"""
GPU uniform-grid collision detection (broadphase + exact narrowphase).

Why this replaces the old version
---------------------------------
1. The old code built an N x N matrix anyway (the "hash" only masked it), so it was
   O(N^2) in time AND memory (several N*N*2 int64/float tensors) and never scaled.
2. Bug: it only treated cells with Manhattan distance <= 1 as neighbours. Cells that
   touch diagonally have Manhattan distance 2, so collisions across a cell corner
   were silently missed. The correct neighbourhood is the 3x3 block (Chebyshev <= 1).
3. The kernel-launch / .item() / torch.where pattern produced the "spike then idle"
   GPU usage: lots of tiny ops with host round-trips between them.

How this one works (no Python loops over particles, 3 host syncs total)
-----------------------------------------------------------------------
- cell = floor((p - min) / cell_size), cell_size >= 2 * max_radius, so two touching
  particles are always in the same or adjacent cells.
- Sort particles by cell id. Every cell is then a contiguous slice of the sorted array.
- For each of 5 "half-stencil" neighbour offsets (own cell + E, NW, N, NE) find the slice
  of that neighbour cell with two torch.searchsorted calls. Using only half of the
  neighbours means each pair is generated exactly once (no duplicates, no triu needed).
- Expand (particle, slice) into candidate pairs with repeat_interleave, then do the exact
  distance test. Work is O(N * average_cell_occupancy), memory is O(candidates).
- Candidates are processed in chunks (max_candidates) so a very dense scene cannot OOM.

Returned pair indices refer to the ORIGINAL particle order, with i != j and each pair once.
"""
from typing import Optional, Tuple

import torch

# (dx, dy) half stencil: own cell handled separately via j > i
_HALF_STENCIL = ((0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def spatial_hash_collisions(
    positions: torch.Tensor,
    radii: torch.Tensor,
    cell_size: Optional[float] = None,
    max_candidates: int = 8_000_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: (N, 2) float tensor (any device).
        radii:     (N,) float tensor, same device.
        cell_size: optional override. Must be >= 2 * radii.max() or collisions are missed.
        max_candidates: soft cap on candidate pairs held in memory at once.

    Returns:
        pairs:    (M, 2) int64 tensor of colliding particle indices (original order)
        overlaps: (M,)   tensor, (r_i + r_j) - distance   (> 0 for every returned pair)
    """
    n = positions.shape[0]
    device = positions.device
    empty = (
        torch.empty((0, 2), dtype=torch.long, device=device),
        torch.empty(0, dtype=positions.dtype, device=device),
    )
    if n < 2:
        return empty

    # Keep cell_size as a 0-d tensor -> no GPU->CPU sync just to compute it.
    cs = (2.0 * radii.max()) if cell_size is None else torch.as_tensor(
        cell_size, dtype=positions.dtype, device=device
    )

    # +1 offset so that "-1" neighbours never go negative; width +3 so +1 never wraps a row.
    mins = positions.min(dim=0).values
    coords = torch.floor((positions - mins) / cs).long() + 1
    width = coords[:, 0].max() + 3
    cell_id = coords[:, 1] * width + coords[:, 0]

    sorted_id, order = torch.sort(cell_id)
    sp = positions[order]
    sr = radii[order]

    idx = torch.arange(n, device=device)

    los, counts = [], []
    for dx, dy in _HALF_STENCIL:
        nid = sorted_id + (dy * width + dx)
        lo = torch.searchsorted(sorted_id, nid, right=False)
        hi = torch.searchsorted(sorted_id, nid, right=True)
        if dx == 0 and dy == 0:
            lo = torch.maximum(lo, idx + 1)  # own cell: only partners after me -> each pair once
        los.append(lo)
        counts.append((hi - lo).clamp(min=0))

    lo_all = torch.cat(los)                    # (5N,) start of candidate slice
    cnt_all = torch.cat(counts)                # (5N,) slice length
    src_all = idx.repeat(len(_HALF_STENCIL))   # (5N,) sorted index of the source particle

    cum = torch.cumsum(cnt_all, dim=0)
    starts = cum - cnt_all

    # --- host sync #1: how many candidates and where to cut chunks ---------------------
    total = int(cum[-1].item())
    if total == 0:
        return empty
    n_chunks = (total + max_candidates - 1) // max_candidates
    if n_chunks == 1:
        bounds = [0, cnt_all.shape[0]]
    else:
        cut_points = torch.arange(0, total, max_candidates, device=device)
        bounds = torch.searchsorted(starts, cut_points, right=False).tolist() + [cnt_all.shape[0]]

    out_pairs, out_overlaps = [], []
    for b in range(len(bounds) - 1):
        a, z = bounds[b], bounds[b + 1]
        if a >= z:
            continue
        c_cnt = cnt_all[a:z]
        c_lo = lo_all[a:z]
        c_src = src_all[a:z]

        m = int(c_cnt.sum().item()) if n_chunks > 1 else total  # (extra sync only if chunked)
        if m == 0:
            continue
        local_start = torch.cumsum(c_cnt, dim=0) - c_cnt
        owner = torch.repeat_interleave(torch.arange(z - a, device=device), c_cnt, output_size=m)

        i = c_src[owner]
        j = torch.arange(m, device=device) - local_start[owner] + c_lo[owner]

        d = sp[i] - sp[j]
        d2 = (d * d).sum(dim=1)
        rs = sr[i] + sr[j]
        hit = d2 < rs * rs                       # exact narrowphase (no sqrt for rejects)

        # --- host sync #2: boolean-mask compaction ------------------------------------
        hi_, hj_ = i[hit], j[hit]
        out_pairs.append(torch.stack((order[hi_], order[hj_]), dim=1))
        out_overlaps.append(rs[hit] - torch.sqrt(d2[hit]))

    if not out_pairs:
        return empty
    return torch.cat(out_pairs), torch.cat(out_overlaps)


if __name__ == "__main__":
    import time

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)

    def brute(pos, rad):
        d = torch.cdist(pos, pos)
        rs = rad[:, None] + rad[None, :]
        iu = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
        ii, jj = torch.where((d < rs) & iu)
        return {(int(a), int(b)) for a, b in zip(ii.tolist(), jj.tolist())}

    # correctness vs brute force, including dense clusters (diagonal-cell case)
    for n, span in [(50, 10.0), (500, 30.0), (2000, 60.0)]:
        pos = (torch.rand(n, 2, generator=g) * span).to(dev)
        rad = (torch.rand(n, generator=g) * 0.4 + 0.1).to(dev)
        pairs, ov = spatial_hash_collisions(pos, rad)
        got = {tuple(sorted(p)) for p in pairs.tolist()}
        ref = brute(pos, rad)
        assert got == ref and len(got) == len(pairs), (n, len(got), len(ref))
        assert bool((ov > 0).all())
        print(f"n={n}: {len(ref)} collisions, matches brute force")

    # speed
    n = 100_000
    pos = (torch.rand(n, 2, generator=g) * 2000).to(dev)
    rad = (torch.rand(n, generator=g) * 0.15 + 0.05).to(dev)
    spatial_hash_collisions(pos, rad)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    p, _ = spatial_hash_collisions(pos, rad)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"{n} particles: {(time.time() - t) * 1e3:.1f} ms, {len(p)} collisions on {dev}")