import torch
import numpy as np
from typing import Tuple, Optional


def spatial_hash_collisions(
    positions: torch.Tensor,
    radii: torch.Tensor,
    cell_size: Optional[float] = None,
    batch_size: int = 2000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-optimized spatial hash collision detection with automatic batching.
    
    Args:
        positions: (N, 2) torch.Tensor of particle positions (must be on device, float32)
        radii: (N,) torch.Tensor of particle radii (must be on device, float32)
        cell_size: Size of hash grid cells (default: 2 * max_radius)
        batch_size: Threshold for switching to batched mode (default: 2000)
    
    Returns:
        collision_pairs: (M, 2) tensor of colliding particle indices
        collision_overlaps: (M,) tensor of overlap distances
        
    Where M is the actual number of collisions found (no padding).
        
    Example:
        # Convert numpy to torch and move to GPU
        positions_t = torch.from_numpy(positions).float().cuda()
        radii_t = torch.from_numpy(radii).float().cuda()
        
        # Run collision detection
        pairs, overlaps = spatial_hash_collisions(positions_t, radii_t)
        print(f"Found {len(pairs)} collisions")
    """
    n = positions.shape[0]
    
    # Choose implementation based on particle count
    if n <= batch_size:
        return _spatial_hash_full(positions, radii, cell_size)
    else:
        return _spatial_hash_batched(positions, radii, cell_size, batch_size)


def _spatial_hash_full(
    positions: torch.Tensor,
    radii: torch.Tensor,
    cell_size: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full matrix version - faster for smaller particle counts (<2000).
    Uses vectorized operations for maximum GPU efficiency.
    """
    n = positions.shape[0]
    device = positions.device
    
    if cell_size is None:
        cell_size = 2.0 * radii.max().item()
    
    # Compute grid coordinates
    grid_coords = torch.floor(positions / cell_size).long()
    
    # Hash cell coordinates using large primes
    PRIME_X = 73856093
    PRIME_Y = 19349663
    cell_hashes = grid_coords[:, 0] * PRIME_X + grid_coords[:, 1] * PRIME_Y
    
    # Sort particles by cell hash for better cache locality
    sort_idx = torch.argsort(cell_hashes)
    sorted_positions = positions[sort_idx]
    sorted_radii = radii[sort_idx]
    sorted_grid_coords = grid_coords[sort_idx]
    
    # Create inverse mapping for results
    inverse_sort = torch.empty(n, dtype=torch.long, device=device)
    inverse_sort[sort_idx] = torch.arange(n, device=device)
    
    # Vectorized collision detection
    # Check if particles are in nearby cells (Manhattan distance <= 1)
    grid_diff = torch.abs(
        sorted_grid_coords.unsqueeze(1) - sorted_grid_coords.unsqueeze(0)
    )
    grid_manhattan = grid_diff.sum(dim=2)
    in_nearby_cells = grid_manhattan <= 1
    
    # Compute pairwise distances
    pos_diff = sorted_positions.unsqueeze(1) - sorted_positions.unsqueeze(0)
    distances = torch.norm(pos_diff, dim=2)
    radii_sum = sorted_radii.unsqueeze(1) + sorted_radii.unsqueeze(0)
    
    # Find collisions (distance < sum of radii AND in nearby cells)
    colliding = (distances < radii_sum) & in_nearby_cells
    
    # Only keep upper triangle to avoid duplicate pairs
    i_indices, j_indices = torch.triu_indices(n, n, offset=1, device=device)
    colliding_upper = colliding[i_indices, j_indices]
    
    # Get collision indices (no padding!)
    collision_indices = torch.where(colliding_upper)[0]
    
    if collision_indices.shape[0] == 0:
        # No collisions - return empty tensors
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty(0, device=device)
        )
    
    collision_i_sorted = i_indices[collision_indices]
    collision_j_sorted = j_indices[collision_indices]
    
    # Map back to original indices
    collision_i = inverse_sort[collision_i_sorted]
    collision_j = inverse_sort[collision_j_sorted]
    collision_pairs = torch.stack([collision_i, collision_j], dim=1)
    
    # Compute overlap distances
    collision_dists = distances[collision_i_sorted, collision_j_sorted]
    collision_overlaps = radii_sum[collision_i_sorted, collision_j_sorted] - collision_dists
    
    return collision_pairs, collision_overlaps


def _spatial_hash_batched(
    positions: torch.Tensor,
    radii: torch.Tensor,
    cell_size: Optional[float] = None,
    batch_size: int = 500
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched version for large particle counts (>2000).
    Processes particles in chunks to manage memory efficiently.
    """
    n = positions.shape[0]
    device = positions.device
    
    if cell_size is None:
        cell_size = 2.0 * radii.max().item()
    
    # Compute grid coordinates and hash
    grid_coords = torch.floor(positions / cell_size).long()
    PRIME_X = 73856093
    PRIME_Y = 19349663
    cell_hashes = grid_coords[:, 0] * PRIME_X + grid_coords[:, 1] * PRIME_Y
    
    # Sort particles by cell hash
    sort_idx = torch.argsort(cell_hashes)
    sorted_positions = positions[sort_idx]
    sorted_radii = radii[sort_idx]
    sorted_grid_coords = grid_coords[sort_idx]
    
    inverse_sort = torch.empty(n, dtype=torch.long, device=device)
    inverse_sort[sort_idx] = torch.arange(n, device=device)
    
    # Collect all collisions
    all_pairs_i = []
    all_pairs_j = []
    all_overlaps = []
    
    # Process in batches to avoid OOM
    n_batches = (n + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n)
        
        batch_positions = sorted_positions[start_idx:end_idx]
        batch_radii = sorted_radii[start_idx:end_idx]
        batch_grid_coords = sorted_grid_coords[start_idx:end_idx]
        
        # Check collisions with all subsequent particles
        remaining_positions = sorted_positions[end_idx:]
        remaining_radii = sorted_radii[end_idx:]
        remaining_grid_coords = sorted_grid_coords[end_idx:]
        
        if remaining_positions.shape[0] > 0:
            # Vectorized grid distance check
            grid_diff = torch.abs(
                batch_grid_coords.unsqueeze(1) - remaining_grid_coords.unsqueeze(0)
            )
            grid_manhattan = grid_diff.sum(dim=2)
            in_nearby = grid_manhattan <= 1
            
            # Vectorized distance check
            pos_diff = batch_positions.unsqueeze(1) - remaining_positions.unsqueeze(0)
            distances = torch.norm(pos_diff, dim=2)
            radii_sum = batch_radii.unsqueeze(1) + remaining_radii.unsqueeze(0)
            
            # Find collisions
            colliding = (distances < radii_sum) & in_nearby
            batch_i, remaining_j = torch.where(colliding)
            
            if batch_i.shape[0] > 0:
                global_i = batch_i + start_idx
                global_j = remaining_j + end_idx
                
                all_pairs_i.append(global_i)
                all_pairs_j.append(global_j)
                
                # Compute overlaps for these collisions
                overlap = radii_sum[batch_i, remaining_j] - distances[batch_i, remaining_j]
                all_overlaps.append(overlap)
        
        # Check within-batch collisions
        if batch_positions.shape[0] > 1:
            batch_size_actual = end_idx - start_idx
            grid_diff = torch.abs(
                batch_grid_coords.unsqueeze(1) - batch_grid_coords.unsqueeze(0)
            )
            grid_manhattan = grid_diff.sum(dim=2)
            in_nearby = grid_manhattan <= 1
            
            pos_diff = batch_positions.unsqueeze(1) - batch_positions.unsqueeze(0)
            distances = torch.norm(pos_diff, dim=2)
            radii_sum = batch_radii.unsqueeze(1) + batch_radii.unsqueeze(0)
            
            colliding = (distances < radii_sum) & in_nearby
            
            # Upper triangle only
            i_idx, j_idx = torch.triu_indices(batch_size_actual, batch_size_actual, 
                                             offset=1, device=device)
            colliding_upper = colliding[i_idx, j_idx]
            valid_within = torch.where(colliding_upper)[0]
            
            if valid_within.shape[0] > 0:
                batch_i = i_idx[valid_within]
                batch_j = j_idx[valid_within]
                global_i = batch_i + start_idx
                global_j = batch_j + start_idx
                
                all_pairs_i.append(global_i)
                all_pairs_j.append(global_j)
                
                overlap = radii_sum[batch_i, batch_j] - distances[batch_i, batch_j]
                all_overlaps.append(overlap)
    
    # Combine results
    if len(all_pairs_i) == 0:
        # No collisions found - return empty tensors
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty(0, device=device)
        )
    
    pairs_i_sorted = torch.cat(all_pairs_i)
    pairs_j_sorted = torch.cat(all_pairs_j)
    overlaps = torch.cat(all_overlaps)
    
    # Map back to original indices
    pairs_i = inverse_sort[pairs_i_sorted]
    pairs_j = inverse_sort[pairs_j_sorted]
    
    collision_pairs = torch.stack([pairs_i, pairs_j], dim=1)
    
    return collision_pairs, overlaps


# Example usage
if __name__ == "__main__":
    print("=== Testing with 100 particles (uses full method) ===")
    np.random.seed(42)
    n_particles = 100
    positions_np = np.random.rand(n_particles, 2).astype(np.float32) * 100.0 - 50.0
    radii_np = np.random.rand(n_particles).astype(np.float32) * 0.3 + 0.1
    
    # Convert to PyTorch tensors and move to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    positions = torch.from_numpy(positions_np).to(device)
    radii = torch.from_numpy(radii_np).to(device)
    
    pairs, overlaps = spatial_hash_collisions(positions, radii)
    
    print(f"Found {len(pairs)} collisions")
    if len(pairs) > 0:
        print(f"First 5 collision pairs:\n{pairs[:5]}")
        print(f"First 5 overlap distances:\n{overlaps[:5]}")
    
    # Test with larger dataset
    print("\n=== Testing with 3000 particles (uses batched method) ===")
    n_particles_large = 3000
    positions_np_large = np.random.rand(n_particles_large, 2).astype(np.float32) * 200.0 - 100.0
    radii_np_large = np.random.rand(n_particles_large).astype(np.float32) * 0.2 + 0.05
    
    positions_large = torch.from_numpy(positions_np_large).to(device)
    radii_large = torch.from_numpy(radii_np_large).to(device)
    
    pairs_large, overlaps_large = spatial_hash_collisions(positions_large, radii_large)
    
    print(f"Found {len(pairs_large)} collisions")
    if len(pairs_large) > 0:
        print(f"First 5 collision pairs:\n{pairs_large[:5]}")
        print(f"First 5 overlap distances:\n{overlaps_large[:5]}")
    
    # Test edge case: no collisions
    print("\n=== Testing with widely spaced particles (no collisions) ===")
    n_sparse = 50
    positions_sparse = torch.rand(n_sparse, 2, device=device) * 1000.0
    radii_sparse = torch.ones(n_sparse, device=device) * 0.1
    
    pairs_sparse, overlaps_sparse = spatial_hash_collisions(positions_sparse, radii_sparse)
    print(f"Found {len(pairs_sparse)} collisions (expected 0)")
    
    # Performance test
    print("\n=== Performance test ===")
    import time
    
    n_test = 5000
    positions_np_test = np.random.rand(n_test, 2).astype(np.float32) * 300.0 - 150.0
    radii_np_test = np.random.rand(n_test).astype(np.float32) * 0.15 + 0.05
    
    positions_test = torch.from_numpy(positions_np_test).to(device)
    radii_test = torch.from_numpy(radii_np_test).to(device)
    
    # Warmup
    _ = spatial_hash_collisions(positions_test, radii_test)
    
    # Time it
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    pairs_test, overlaps_test = spatial_hash_collisions(positions_test, radii_test)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    
    print(f"Processed {n_test} particles in {(end-start)*1000:.2f}ms")
    print(f"Found {len(pairs_test)} collisions")
    print(f"Device: {device}")