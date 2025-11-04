#!/usr/bin/env python3
"""
Example script demonstrating batch processing of multiple image pairs using PopSift.

This shows how to efficiently match multiple image pairs using the new batch functions
that reuse a single PopSift instance and enqueue all images at once.
"""

import os
import numpy as np
import popsift_match
import time
from popsift_config import SiftConfig
try:
    import cv2
except ImportError:
    print("Error: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

def load_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img

def flann_matching_cpu(desc1, desc2):
    #> Create matcher and match using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    start = time.time()
    # k=2 nearest neighbors
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    ratio_thresh = 0.7
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    return time.time() - start

def example_batch_from_files(left_files, right_files):
    """Example: Batch process multiple image pairs from files."""
    print("=" * 60)
    print("Example: Batch Processing from Files")
    print("=" * 60)
    
    # Process all pairs at once (efficient!)
    results = popsift_match.match_multiple_pairs_from_files(
        left_files,
        right_files,
        verbose=True,
        print_time_info=True
    )
    
    # Process results
    for i, result in enumerate(results):
        print(f"\nPair {i} Results:")
        print(f"  Number of Accepted Matches: {result.num_matches}")
        print(f"  Left GPU time: {result.left_gpu_time_ms:.2f} ms")
        print(f"  Right GPU time: {result.right_gpu_time_ms:.2f} ms")
        print(f"  Matching time: {result.match_time_ms:.2f} ms")
        if result.num_matches > 0:
            print(f"  Average match distance: {np.mean(result.match_distances):.2f}")


def example_batch_from_arrays(left_image_files, right_image_files):
    """Example: Batch process multiple image pairs from numpy arrays."""
    print("\n" + "=" * 60)
    print("Example: Batch Processing from NumPy Arrays")
    print("=" * 60)
    
    left_images = [
        load_img(left_image_files[0]),
        load_img(left_image_files[1]),
        load_img(left_image_files[2]),
        load_img(left_image_files[3])
    ]
    
    right_images = [
        load_img(right_image_files[0]),
        load_img(right_image_files[1]),
        load_img(right_image_files[2]),
        load_img(right_image_files[3])
    ]
    
    # ====================================================================
    sift = cv2.SIFT_create()
    total_flann_match_cpu_time = 0
    for left, right in zip(left_images, right_images):
        kpt1, desc1 = sift.detectAndCompute(left, None)
        kpt2, desc2 = sift.detectAndCompute(right, None)
        total_flann_match_cpu_time += flann_matching_cpu(desc1, desc2)
    print(f"Matching time on CPU: {total_flann_match_cpu_time} seconds")
    
    # ====================================================================
    print(f"\nMethod 1: Processing each image pair at a time")
    start = time.time()
    results_single = []
    for left, right in zip(left_images, right_images):
        result = popsift_match.match_features_from_arrays(left, right, verbose=False)
        results_single.append(result)
    time_single = time.time() - start
    
    # ====================================================================
    print(f"Method 2: Batch processing all image pairs")
    start = time.time()
    results_batch = popsift_match.match_multiple_pairs_from_arrays(
        left_images,
        right_images,
        verbose=False,
        print_time_info=False
    )
    time_batch = time.time() - start
    
    # ====================================================================
    print(f"\nResults:")
    print(f"  Single processing: {time_single:.3f} seconds")
    print(f"  Batch processing:  {time_batch:.3f} seconds")
    print(f"  Speedup: {time_single/time_batch:.2f}x faster")
    
    # Process results
    for i, result in enumerate(results_single):
        print(f"\nPair {i} Results (Single):")
        print(f"  Matches: {result.num_matches}")
        print(f"  Total matches (before filtering): {result.num_total_matches}")
        print(f"  Left GPU time: {result.left_gpu_time_ms:.2f} ms")
        print(f"  Right GPU time: {result.right_gpu_time_ms:.2f} ms")
        print(f"  Matching time: {result.match_time_ms:.2f} ms")
    for i, result in enumerate(results_batch):
        print(f"\nPair {i} Results (Batch):")
        print(f"  Matches: {result.num_matches}")
        print(f"  Total matches (before filtering): {result.num_total_matches}")
        print(f"  Left GPU time: {result.left_gpu_time_ms:.2f} ms")
        print(f"  Right GPU time: {result.right_gpu_time_ms:.2f} ms")
        print(f"  Matching time: {result.match_time_ms:.2f} ms")
        print(f"  Matching time: {result.total_batch_time_ms:.2f} ms")


def example_batch_with_custom_config():
    """Example: Batch process with custom SIFT configuration."""
    print("\n" + "=" * 60)
    print("Example: Batch Processing with Custom Configuration")
    print("=" * 60)
    
    # Create custom SIFT configuration
    config = SiftConfig()
    config.octaves = 4
    config.levels = 3
    config.threshold = 0.01
    
    # Create dummy images
    left_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    right_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    # Process with custom config
    results = popsift_match.match_multiple_pairs_from_arrays_with_config(
        left_images,
        right_images,
        config,
        verbose=True,
        print_time_info=True
    )
    
    print(f"\nProcessed {len(results)} pairs with custom configuration")
    for i, result in enumerate(results):
        print(f"Pair {i}: {result.num_matches} matches")

def comparison_single_vs_batch():
    """Compare performance: processing pairs one-by-one vs batch."""
    print("\n" + "=" * 60)
    print("Performance Comparison: Single vs Batch Processing")
    print("=" * 60)

    # Create test images
    num_pairs = 5
    left_images = [np.random.randint(0, 256, (480, 640), dtype=np.uint8) for _ in range(num_pairs)]
    right_images = [np.random.randint(0, 256, (480, 640), dtype=np.uint8) for _ in range(num_pairs)]
    
    # Method 1: Process one pair at a time
    print(f"\nMethod 1: Processing {num_pairs} pairs one at a time...")
    start = time.time()
    results_single = []
    for left, right in zip(left_images, right_images):
        result = popsift_match.match_features_from_arrays(left, right, verbose=False)
        results_single.append(result)
    time_single = time.time() - start
    
    # Method 2: Batch process all pairs
    print(f"Method 2: Batch processing {num_pairs} pairs...")
    start = time.time()
    results_batch = popsift_match.match_multiple_pairs_from_arrays(
        left_images, 
        right_images, 
        verbose=False
    )
    time_batch = time.time() - start
    
    # Compare
    print(f"\nResults:")
    print(f"  Single processing: {time_single:.3f} seconds")
    print(f"  Batch processing:  {time_batch:.3f} seconds")
    print(f"  Speedup: {time_single/time_batch:.2f}x faster")
    print(f"\n  Note: Speedup comes from:")
    print(f"  - Reusing single PopSift instance (no repeated initialization)")
    print(f"  - Pipeline keeps GPU busy while uploading next images")
    print(f"  - Reduced CPU-GPU synchronization overhead")


def main():
    """Run all examples."""
    print("PopSift Batch Matching Examples")
    print(f"Version: {popsift_match.get_version()}\n")
    
    # If you have actual image files, uncomment this:
    dataset_path = "/home/jetsonlems/"
    
    left_files = [
        os.path.join(dataset_path, "hetlf_326766299.jpg"),
        os.path.join(dataset_path, "hetlf_326766299.jpg"),
        os.path.join(dataset_path, "hetlf_326766299.jpg"),
        os.path.join(dataset_path, "hetlf_326766299.jpg"),
    ]
    right_files = [
        os.path.join(dataset_path, "hetrf_326766299.jpg"),
        os.path.join(dataset_path, "hetrf_326766299.jpg"),
        os.path.join(dataset_path, "hetrf_326766299.jpg"),
        os.path.join(dataset_path, "hetrf_326766299.jpg"),
    ]

    # example_batch_from_files(left_files, right_files)
    
    example_batch_from_arrays(left_files, right_files)
    # example_batch_with_custom_config()
    
    # Performance comparison (works with dummy synthetic data):
    #comparison_single_vs_batch()


if __name__ == "__main__":
    main()

