import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable
import time
import random

# ============================================================================
# SOC SANDPILE CORE
# ============================================================================

def relax_sandpile_2d(heights: np.ndarray,
                     threshold: int = 4,
                     grains_per_neighbor: int = 1) -> Tuple[np.ndarray, int]:
    """Relax 2D sandpile, return avalanche map and size."""
    H, W = heights.shape
    avalanche_map = np.zeros_like(heights, dtype=int)
    grains_lost = 4 * grains_per_neighbor
    
    while True:
        unstable = heights >= threshold
        if not np.any(unstable):
            break
        
        toppled = unstable.astype(int)
        avalanche_map += toppled
        heights -= toppled * grains_lost
        
        heights[:-1, :] += toppled[1:, :] * grains_per_neighbor
        heights[1:, :] += toppled[:-1, :] * grains_per_neighbor
        heights[:, :-1] += toppled[:, 1:] * grains_per_neighbor
        heights[:, 1:] += toppled[:, :-1] * grains_per_neighbor
    
    return avalanche_map, int(avalanche_map.sum())


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_sandpile_grids(mask_size: Tuple[int, int]) -> np.ndarray:
    H, W = mask_size
    return np.zeros((H, W), dtype=int)

# ============================================================================
# THERMALIZATION
# ============================================================================

def thermalize_sandpiles(sandpile_grid: np.ndarray,
                        threshold: int = 4,
                        grains_per_neighbor: int = 1,
                        n_drops: int = 50000):
    """Drive sandpiles to critical state."""
    
    print("Thermalizing sandpiles to critical state...")
    for _ in range(n_drops):
        y = np.random.randint(0, sandpile_grid.shape[0])
        x = np.random.randint(0, sandpile_grid.shape[1])
        sandpile_grid[y, x] += 1
        relax_sandpile_2d(sandpile_grid, threshold, grains_per_neighbor)
        
    print(f"Mean height = {np.mean(sandpile_grid):.2f}")


# ============================================================================
# AVALANCHE GENERATION
# ============================================================================

def generate_avalanche_pattern(sandpile_grid: np.ndarray,
                              threshold: int = 4,
                              grains_per_neighbor: int = 1) -> Tuple[np.ndarray, int]:
    """Trigger single avalanche, return avalanche map."""
    
    H, W = sandpile_grid.shape
    y = np.random.randint(0, H)
    x = np.random.randint(0, W)
    sandpile_grid[y, x] += 1
    
    return relax_sandpile_2d(sandpile_grid, threshold, grains_per_neighbor)


# ============================================================================
# PERTURBATION AND MASK UPDATE (UINT8)
# ============================================================================

def perturb_mask_uint8(mask: np.ndarray,
                      avalanche_map: np.ndarray,
                      perturbation_scale: int,
                      avalanche_size: int) -> np.ndarray:
    
    trial_mask = (mask.copy()).astype(np.float32)  # Use int16 for intermediate calculations
    
    # Adaptive scale based on avalanche size
    #adaptive_scale = perturbation_scale * np.log10(max(2, avalanche_size))
    adaptive_scale = perturbation_scale
    # Perturbation at avalanche sites only
    mean = 0
    std = adaptive_scale
    perturbation = np.random.normal(mean, std, mask.shape) 
    perturbation *= (avalanche_map > 0)
    
    # Apply perturbation
    trial_mask = trial_mask + perturbation.astype(np.int16)
    
    # Wrap to [0, 255]
    trial_mask = trial_mask % 256
    trial_mask = np.clip(trial_mask, 0, 255)
    
    return trial_mask.astype(np.uint8)

# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

def soc_optimize_experimental(
    slm, data,
    measure_accuracy_fn: Callable,
    upload_masks_fn: Callable,
    initial_mask: np.ndarray,
    mask_size: Tuple[int, int] = (50, 50),
    max_iterations: int = 2000,
    threshold: int = 4,
    grains_per_neighbor: int = 1,
    perturbation_scale: int = 50,
    verbose: bool = True,
    save_every: int = 50,
    random_seed: int = 42) -> Dict: 
    
    print(f"Setting random seed to {random_seed} for reproducibility...")
    np.random.seed(random_seed)
    random.seed(random_seed)  # Also seed Python's random module
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    current_mask = initial_mask.copy()
    sandpile_grid = initialize_sandpile_grids(mask_size)
    
    accuracy_history = []
    avalanche_size_history = []
    mask_history = []
    best_accuracy = -np.inf
    best_mask = None
    
    # ========================================================================
    # THERMALIZATION
    # ========================================================================
    
    thermalize_sandpiles(sandpile_grid, threshold, grains_per_neighbor, 
                        n_drops=150*150*3)
    out_imgs, labels = upload_masks_fn(
            slm, data, current_mask, [0,1],1
    )
        
    current_accuracy = measure_accuracy_fn(out_imgs, labels)
    
    accuracy_history.append(current_accuracy)
    mask_history.append(current_mask)
    best_accuracy = current_accuracy
    best_mask = current_mask.copy()
    
    filename = f"not_optimized_21_11_25_day_flowers_start.npz"
    np.savez(filename, out_imgs = out_imgs, labels=labels,mask=best_mask )
    
    if verbose:
        print(f"\nInitial accuracy: {current_accuracy:.4f} ")
        print(f"--- Starting SOC optimization ({max_iterations} iterations) ---\n")
    
    start_time = time.time()
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    for iteration in range(max_iterations):
            
        q_data_idx = np.random.choice([0, 1], size=2, replace=False)
        while True:
            avalanche_map, avalanche_size = generate_avalanche_pattern(
                sandpile_grid, threshold, grains_per_neighbor
            )
            if avalanche_size < 1:
                continue
            else:
                break

        trial_mask = perturb_mask_uint8(
            current_mask, avalanche_map, 
            perturbation_scale, avalanche_size
            )
                
        out_1, labels_1 = upload_masks_fn(
            slm,data,trial_mask,[q_data_idx[0]],1
            )
        
        # Measure accuracy
        print(f"iteration {iteration}: Q1r \n Avalanche size : {avalanche_size}")
        trial_accuracy_1 = measure_accuracy_fn(out_1, labels_1)
        
        
        # Accept/reject
        if trial_accuracy_1 > (current_accuracy - 0.03):
            out_2, labels_2 = upload_masks_fn(
                slm,data,trial_mask,[q_data_idx[1]],1
                )

            out_imgs = np.concatenate([out_1, out_2], axis=0)
            labels = np.concatenate([labels_1, labels_2], axis=0)
            
                # Measure accuracy
            print(f"iteration {iteration}: Q2r \n Avalanche size : {avalanche_size}")
            trial_accuracy_12 = measure_accuracy_fn(out_imgs, labels)
            
            if trial_accuracy_12 > (current_accuracy - 0.01):
                current_mask = trial_mask.copy()
                current_accuracy = trial_accuracy_12
                best_accuracy = trial_accuracy_12
                best_mask = trial_mask.copy()
                accuracy_history.append(current_accuracy)
                mask_history.append(current_mask)
                avalanche_size_history.append(avalanche_size)
                if verbose:
                    print(f"*** New best: {best_accuracy:.4f} at iteration {iteration+1}")

        else:
            accuracy_history.append(trial_accuracy_1)
            mask_history.append(trial_mask)
            avalanche_size_history.append(avalanche_size)

        
        # Save masks periodically
        if (iteration + 1) % save_every == 0:
            filename = f"masks_iter_{iteration+1}21_11_25_day_flowers.npz"
            np.savez(filename, avalanche_size_history=avalanche_size_history,
                     mask_history=mask_history, accuracy_history=accuracy_history,
                     best_mask=best_mask)
            if verbose:
                print(f"Saved: {filename}")

    # ========================================================================
    # FINALIZATION
    # ========================================================================
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n--- Optimization complete ---")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"\n Re-running Q1234 with optimized mask:")
        out_img,labels = upload_masks_fn(
            slm, data, best_mask, [0,1],1
        )
        
        accuracy_best_mask = measure_accuracy_fn(out_imgs,labels)
        print(f"Accuracy best mask: {accuracy_best_mask:.4f}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        
        filename = f"optimized_21_11_25_day_flowers.npz"
        np.savez(filename, out_imgs=out_imgs, labels=labels,mask=best_mask )
        

        print(f"\n Re-running Q1234 with initial mask:")
        out_imgs, labels = upload_masks_fn(
            slm, data, initial_mask, [0,1],1
        )
        
        accuracy_initial_mask = measure_accuracy_fn(out_imgs,labels)
        print(f"Accuracy initial mask: {accuracy_initial_mask:.4f}")
        
        filename = f"not_optimized_21_11_25_day_flowers_end.npz"
        np.savez(filename, out_imgs=out_imgs, labels=labels,mask=initial_mask )
    
    return {
        'best_mask': best_mask,
        'best_accuracy': best_accuracy,
        'accuracy_best_mask': accuracy_best_mask,
        'accuracy_history': accuracy_history,
        'avalanche_size_history': avalanche_size_history,
        'initial_mask': initial_mask,
        'mask_history':mask_history
    }


# ============================================================================
# UTILITIES - FIXED
# ============================================================================

def save_masks_uint8(best_mask, best_accuracy, accuracy_best_mask, 
                    accuracy_history, avalanche_size_history, 
                    initial_mask,mask_history,
                    filename: str):
    """Save uint8 phase masks to file."""
    np.savez(filename, 
             mask=best_mask, 
             accuracy=best_accuracy,
             accuracy_best_mask=accuracy_best_mask,
             accuracy_history=accuracy_history,
             avalanche_size_history=avalanche_size_history,
             initial_mask=initial_mask,
             mask_history=mask_history)
    print(f"Saved: {filename}")


def load_masks_uint8(filename: str) -> Tuple[np.ndarray, float]:
    """Load uint8 phase masks from file."""
    data = np.load(filename, allow_pickle=True)
    mask = data['mask'].astype(np.uint8)
    accuracy = float(data['accuracy'])
    print(f"Loaded from {filename}: accuracy = {accuracy:.4f}")
    return mask, accuracy
