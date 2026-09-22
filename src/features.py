import cv2
import numpy as np
from pathlib import Path

def extract_spatial_features(image):
    """Extracts mean, std, and color histograms across RGB channels."""
    # Convert image to float32 for statistical calculations
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))
    
    # Calculate RGB Color Histograms (16 bins per channel)
    hist_r = cv2.calcHist([image], [0], None, [16], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [16], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [16], [0, 256]).flatten()
    
    # Normalize histograms
    hist_r /= np.sum(hist_r) + 1e-7
    hist_g /= np.sum(hist_g) + 1e-7
    hist_b /= np.sum(hist_b) + 1e-7
    
    return np.hstack([mean_color, std_color, hist_r, hist_g, hist_b])

def extract_fft_features(gray_image):
    """Extracts 2D Fast Fourier Transform (FFT) frequency magnitude spectrum features."""
    # Compute 2D Discrete Fourier Transform
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    
    # Magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-7)
    
    # Flatten FFT magnitude representation or summarize radial distributions
    fft_mean = np.mean(magnitude_spectrum)
    fft_std = np.std(magnitude_spectrum)
    fft_max = np.max(magnitude_spectrum)
    
    return np.array([fft_mean, fft_std, fft_max])

def extract_single_image_features(image_path):
    """Combines spatial and frequency domain features for one image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    spatial_feat = extract_spatial_features(img_rgb)
    fft_feat = extract_fft_features(img_gray)
    
    return np.hstack([spatial_feat, fft_feat])

def process_batch(base_dir, split='train', max_samples_per_class=1000):
    """
    Loads images from base_dir/split/(REAL|FAKE) and extracts feature vectors.
    Returns: X (feature matrix), y (binary labels: 0 for REAL, 1 for FAKE)
    """
    base_path = Path(base_dir) / split
    real_path = base_path / 'REAL'
    fake_path = base_path / 'FAKE'
    
    X, y = [], []
    
    # Process REAL images (Label 0)
    real_images = list(real_path.glob('*.jpg')) + list(real_path.glob('*.png'))
    for img_path in real_images[:max_samples_per_class]:
        feat = extract_single_image_features(img_path)
        if feat is not None:
            X.append(feat)
            y.append(0)
            
    # Process FAKE images (Label 1)
    fake_images = list(fake_path.glob('*.jpg')) + list(fake_path.glob('*.png'))
    for img_path in fake_images[:max_samples_per_class]:
        feat = extract_single_image_features(img_path)
        if feat is not None:
            X.append(feat)
            y.append(1)
            
    return np.array(X), np.array(y)