import numpy as np
import scipy.spatial as ss
# from numba import njit

def detect_collisions(positions):
    tree = ss.cKDTree(positions) # pyright: ignore[reportAttributeAccessIssue]
    collision_radius = 0.05
    # Find all pairs within collision radius
    return tree.query_pairs(r=collision_radius)



def merge_kernel(pairs_i, pairs_j, masses, positions, velocities):
    N = masses.shape[0]
    alive = masses > 0
    has_won = np.zeros(N, dtype=np.bool_)  # Track planets that have already won

    winners = np.empty(len(pairs_i), dtype=np.int32)
    losers = np.empty(len(pairs_i), dtype=np.int32)
    count = 0

    for k in range(len(pairs_i)):
        i = pairs_i[k]
        j = pairs_j[k]

        if not (alive[i] and alive[j]):
            continue  # skip if either is already dead

        # Determine winner: prefer the one that has already won, otherwise use mass
        if has_won[i] and not has_won[j]:
            w, l = i, j
        elif has_won[j] and not has_won[i]:
            w, l = j, i
        elif masses[i] >= masses[j]:
            w, l = i, j
        else:
            w, l = j, i

        # merge
        mw = masses[w]
        ml = masses[l]
        M = mw + ml

        velocities[w, 0] = (mw * velocities[w, 0] + ml * velocities[l, 0]) / M
        velocities[w, 1] = (mw * velocities[w, 1] + ml * velocities[l, 1]) / M

        positions[w, 0] = (mw * positions[w, 0] + ml * positions[l, 0]) / M
        positions[w, 1] = (mw * positions[w, 1] + ml * positions[l, 1]) / M

        masses[w] = M

        # kill loser
        masses[l] = 0.0
        alive[l] = False
        has_won[w] = True  # Mark winner so it can't lose later

        winners[count] = w
        losers[count] = l
        count += 1

    return winners[:count], losers[:count]