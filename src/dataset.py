"""
src/dataset.py
Dataset verification, shape inspection, and sample visualization module for CIFAKE.
"""

from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Base Data Directory Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def verify_dataset_counts(data_dir: Path = DATA_DIR) -> bool:
    """
    Verifies image counts across train and test splits for REAL and FAKE classes.
    Expected:
      - train/REAL: 50,000 | train/FAKE: 50,000
      - test/REAL:  10,000 | test/FAKE:  10,000
    """
    splits = {
        "train": {"REAL": 50000, "FAKE": 50000},
        "test": {"REAL": 10000, "FAKE": 10000}
    }
    
    print("\n" + "=" * 50)
    print(" 1. DATASET IMAGE COUNT VERIFICATION ")
    print("=" * 50)
    
    all_valid = True
    for split, categories in splits.items():
        for category, expected_count in categories.items():
            folder_path = data_dir / split / category
            if not folder_path.exists():
                print(f"❌ Missing directory: {folder_path}")
                all_valid = False
                continue
            
            # Count valid image files (.jpg, .png, .jpeg)
            images = [f for f in folder_path.glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            actual_count = len(images)
            
            status = "✅ PASS" if actual_count == expected_count else f"⚠️ MISMATCH (Found {actual_count})"
            print(f"[{split.upper()}] {category:<4}: {actual_count:,} / {expected_count:,} images -> {status}")
            
            if actual_count != expected_count:
                all_valid = False

    return all_valid


def verify_image_properties(data_dir: Path = DATA_DIR):
    """
    Checks shape, data type, and min/max pixel values of a sample image.
    """
    print("\n" + "=" * 50)
    print(" 2. IMAGE PROPERTIES VERIFICATION ")
    print("=" * 50)
    
    sample_path = next((data_dir / "train" / "REAL").glob("*.jpg"), None)
    if not sample_path:
        sample_path = next((data_dir / "train" / "REAL").glob("*.png"), None)
        
    if not sample_path:
        print("❌ No sample image found. Ensure extraction is complete.")
        return

    # Read image using OpenCV (BGR format)
    img_bgr = cv2.imread(str(sample_path))
    
    print(f"Sample File:       {sample_path.name}")
    print(f"Shape (H x W x C): {img_bgr.shape}")
    print(f"Data Type:         {img_bgr.dtype}")
    print(f"Pixel Range:       Min = {img_bgr.min()}, Max = {img_bgr.max()}")
    
    if img_bgr.shape == (32, 32, 3):
        print("✅ Image shape verified successfully (32 x 32 RGB).")
    else:
        print(f"⚠️ Unexpected shape: {img_bgr.shape}")


def visualize_samples(data_dir: Path = DATA_DIR, save_path: Path = None):
    """
    Displays a 2x5 grid comparing sample Real vs. Fake images.
    """
    print("\n" + "=" * 50)
    print(" 3. GENERATING SAMPLE COMPARISON GRID ")
    print("=" * 50)

    real_paths = list((data_dir / "train" / "REAL").glob("*"))
    fake_paths = list((data_dir / "train" / "FAKE").glob("*"))

    if not real_paths or not fake_paths:
        print("❌ Cannot display samples - image folders are empty.")
        return

    random.seed(42)
    selected_real = random.sample(real_paths, 5)
    selected_fake = random.sample(fake_paths, 5)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("CIFAKE Dataset: Real vs. AI-Generated (Top: Real | Bottom: Fake)", fontsize=13, fontweight="bold")

    for idx in range(5):
        # Read and convert Real Image (BGR -> RGB)
        r_img = cv2.cvtColor(cv2.imread(str(selected_real[idx])), cv2.COLOR_BGR2RGB)
        axes[0, idx].imshow(r_img)
        axes[0, idx].set_title(f"Real #{idx+1}", fontsize=10)
        axes[0, idx].axis("off")

        # Read and convert Fake Image (BGR -> RGB)
        f_img = cv2.cvtColor(cv2.imread(str(selected_fake[idx])), cv2.COLOR_BGR2RGB)
        axes[1, idx].imshow(f_img)
        axes[1, idx].set_title(f"Fake #{idx+1}", fontsize=10)
        axes[1, idx].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved grid to {save_path}")
    plt.show()


if __name__ == "__main__":
    counts_ok = verify_dataset_counts()
    verify_image_properties()
    if counts_ok:
        visualize_samples()
