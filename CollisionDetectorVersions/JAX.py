import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from jax import Array
from functools import partial



def spatial_hash_collisions_optimized(
    positions: Array,
    radii: Array,
    cell_size: float = None,    #type: ignore
    batch_size: int = 2000
) -> Tuple[Array, Array, Array]:
    """
    Unified GPU-optimized spatial hash collision detection with automatic batching.
    Dynamically switches between full and batched computation based on particle count.
    
    Args:
        positions: (N, 2) array of particle positions (can be any range)
        radii: (N,) array of particle radii
        cell_size: Size of hash grid cells (default: 2 * max_radius)
        batch_size: Threshold for switching to batched mode (default: 2000)
    
    Returns:
        collision_pairs: (M, 2) array of colliding particle indices (padded with -1)
        collision_distances: (M,) array of overlap distances (padded with 0)
        valid_mask: (M,) boolean array indicating valid collisions
        
    Note: Filter results using valid_mask to get actual collisions:
        valid_pairs = collision_pairs[valid_mask]
        valid_overlaps = collision_distances[valid_mask]
    """
    n = positions.shape[0]
    
    # Choose implementation based on particle count
    if n <= batch_size:
        return _spatial_hash_full(positions, radii, cell_size)
    else:
        return _spatial_hash_batched(positions, radii, cell_size, batch_size)



@jit
def _spatial_hash_full(
    positions: Array,
    radii: Array,
    cell_size: float = None  # type: ignore
) -> Tuple[Array, Array, Array]:
    """
    Full matrix version - faster for smaller particle counts (<2000).
    """
    n = positions.shape[0]
    max_collisions = n * (n - 1) // 2
    
    if cell_size is None:
        cell_size = 2.0 * jnp.max(radii)
    
    # Compute grid coordinates
    grid_coords = jnp.floor(positions / cell_size).astype(jnp.int32)
    
    # Hash cell coordinates
    PRIME_X = 73856093
    PRIME_Y = 19349663
    cell_hashes = grid_coords[:, 0] * PRIME_X + grid_coords[:, 1] * PRIME_Y
    
    # Sort particles by cell hash
    sort_idx = jnp.argsort(cell_hashes)
    sorted_positions = positions[sort_idx]
    sorted_radii = radii[sort_idx]
    sorted_grid_coords = grid_coords[sort_idx]
    
    # Create inverse mapping
    inverse_sort = jnp.empty(n, dtype=jnp.int32)
    inverse_sort = inverse_sort.at[sort_idx].set(jnp.arange(n))
    
    # Vectorized collision detection
    grid_diff = jnp.abs(sorted_grid_coords[:, None, :] - sorted_grid_coords[None, :, :])
    grid_manhattan = jnp.sum(grid_diff, axis=2)
    in_nearby_cells = grid_manhattan <= 1
    
    pos_diff = sorted_positions[:, None, :] - sorted_positions[None, :, :]
    distances = jnp.linalg.norm(pos_diff, axis=2)
    radii_sum = sorted_radii[:, None] + sorted_radii[None, :]
    
    colliding = (distances < radii_sum) & in_nearby_cells
    
    # Only upper triangle
    i_indices, j_indices = jnp.triu_indices(n, k=1)
    colliding_upper = colliding[i_indices, j_indices]
    
    # Use jnp.where instead of boolean indexing - THIS IS THE KEY FIX
    collision_flat_indices = jnp.where(colliding_upper, size=max_collisions, fill_value=-1)[0]
    valid_mask = collision_flat_indices >= 0
    
    safe_indices = jnp.where(valid_mask, collision_flat_indices, 0)
    collision_i_sorted = i_indices[safe_indices]
    collision_j_sorted = j_indices[safe_indices]
    
    # Map back to original indices
    collision_i = inverse_sort[collision_i_sorted]
    collision_j = inverse_sort[collision_j_sorted]
    collision_pairs = jnp.stack([collision_i, collision_j], axis=1)
    
    collision_dists = distances[collision_i_sorted, collision_j_sorted]
    overlap = jnp.where(valid_mask, 
                        radii_sum[collision_i_sorted, collision_j_sorted] - collision_dists,
                        0.0)
    
    return collision_pairs, overlap, valid_mask  # type: ignore

def _spatial_hash_batched(
    positions: Array,
    radii: Array,
    cell_size: float = None,    #type: ignore
    batch_size: int = 2000
) -> Tuple[Array, Array, Array]:
    """
    Batched version for large particle counts (>2000).
    Processes particles in chunks to avoid GPU OOM.
    """
    n = positions.shape[0]
    
    if cell_size is None:
        cell_size = 2.0 * jnp.max(radii)
    
    # Compute grid coordinates and hash
    grid_coords = jnp.floor(positions / cell_size).astype(jnp.int32)
    PRIME_X = 73856093
    PRIME_Y = 19349663
    cell_hashes = grid_coords[:, 0] * PRIME_X + grid_coords[:, 1] * PRIME_Y
    
    # Sort particles by cell hash
    sort_idx = jnp.argsort(cell_hashes)
    sorted_positions = positions[sort_idx]
    sorted_radii = radii[sort_idx]
    sorted_grid_coords = grid_coords[sort_idx]
    sorted_hashes = cell_hashes[sort_idx]
    
    inverse_sort = jnp.empty(n, dtype=jnp.int32)
    inverse_sort = inverse_sort.at[sort_idx].set(jnp.arange(n))
    
    # Find cell boundaries for efficient neighbor search
    unique_hashes, cell_starts = jnp.unique(sorted_hashes, return_index=True, size=n)
    cell_ends = jnp.concatenate([cell_starts[1:], jnp.array([n])])
    
    # Process in batches
    all_pairs_i = []
    all_pairs_j = []
    all_overlaps = []
    
    n_batches = (n + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n)
        
        batch_positions = sorted_positions[start_idx:end_idx]
        batch_radii = sorted_radii[start_idx:end_idx]
        batch_grid_coords = sorted_grid_coords[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # For each particle in batch, check against all particles in nearby cells
        # This includes particles in later batches (to avoid missing collisions)
        for i in range(batch_size_actual):
            global_i = start_idx + i
            particle_pos = batch_positions[i]
            particle_radius = batch_radii[i]
            particle_grid = batch_grid_coords[i]
            
            # Check all particles that could be in neighboring cells
            # Only check particles with index > global_i to avoid duplicates
            for j in range(global_i + 1, n):
                other_grid = sorted_grid_coords[j]
                
                # Quick grid distance check
                grid_dist = jnp.abs(particle_grid[0] - other_grid[0]) + jnp.abs(particle_grid[1] - other_grid[1])
                
                if grid_dist <= 1:
                    # Actual distance check
                    other_pos = sorted_positions[j]
                    other_radius = sorted_radii[j]
                    
                    dist = jnp.linalg.norm(particle_pos - other_pos)
                    radius_sum = particle_radius + other_radius
                    
                    if dist < radius_sum:
                        all_pairs_i.append(global_i)
                        all_pairs_j.append(j)
                        all_overlaps.append(radius_sum - dist)
    
    # Convert to arrays
    if len(all_pairs_i) == 0:
        # No collisions found
        max_collisions = n * (n - 1) // 2
        return (jnp.full((max_collisions, 2), -1, dtype=jnp.int32),
                jnp.zeros(max_collisions),
                jnp.zeros(max_collisions, dtype=bool))
    
    pairs_i_sorted = jnp.array(all_pairs_i, dtype=jnp.int32)
    pairs_j_sorted = jnp.array(all_pairs_j, dtype=jnp.int32)
    overlaps = jnp.array(all_overlaps)
    
    # Map back to original indices
    pairs_i = inverse_sort[pairs_i_sorted]
    pairs_j = inverse_sort[pairs_j_sorted]
    
    # Pad to standard size
    n_collisions = len(pairs_i)
    max_collisions = n * (n - 1) // 2
    
    collision_pairs = jnp.full((max_collisions, 2), -1, dtype=jnp.int32)
    collision_pairs = collision_pairs.at[:n_collisions, 0].set(pairs_i)
    collision_pairs = collision_pairs.at[:n_collisions, 1].set(pairs_j)
    
    collision_overlaps = jnp.zeros(max_collisions)
    collision_overlaps = collision_overlaps.at[:n_collisions].set(overlaps)
    
    valid_mask = jnp.arange(max_collisions) < n_collisions
    
    return collision_pairs, collision_overlaps, valid_mask


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Test with small dataset
    print("=== Testing with 100 particles (uses full method) ===")
    np.random.seed(42)
    n_particles = 100
    positions = np.random.rand(n_particles, 2) * 100.0 - 50.0
    radii = np.random.rand(n_particles) * 0.3 + 0.1
    
    positions_jax = jnp.array(positions)
    radii_jax = jnp.array(radii)
    
    pairs, overlaps, valid = spatial_hash_collisions_optimized(positions_jax, radii_jax)
    valid_pairs = pairs[valid]
    valid_overlaps = overlaps[valid]
    
    print(f"Found {len(valid_pairs)} collisions")
    if len(valid_pairs) > 0:
        print(f"First 5 collision pairs:\n{valid_pairs[:5]}")
        print(f"First 5 overlap distances:\n{valid_overlaps[:5]}")
    
    # Test with larger dataset
    print("\n=== Testing with 3000 particles (uses batched method) ===")
    n_particles_large = 3000
    positions_large = np.random.rand(n_particles_large, 2) * 200.0 - 100.0
    radii_large = np.random.rand(n_particles_large) * 0.2 + 0.05
    
    positions_jax_large = jnp.array(positions_large)
    radii_jax_large = jnp.array(radii_large)
    
    pairs_large, overlaps_large, valid_large = spatial_hash_collisions_optimized(
        positions_jax_large, radii_jax_large
    )
    valid_pairs_large = pairs_large[valid_large]
    valid_overlaps_large = overlaps_large[valid_large]
    
    print(f"Found {len(valid_pairs_large)} collisions")
    if len(valid_pairs_large) > 0:
        print(f"First 5 collision pairs:\n{valid_pairs_large[:5]}")
        print(f"First 5 overlap distances:\n{valid_overlaps_large[:5]}")