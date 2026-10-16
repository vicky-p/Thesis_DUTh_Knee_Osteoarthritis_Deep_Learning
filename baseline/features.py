"""
Hand-crafted feature extraction reproducing the Thomson 2016 baseline.
~84 features per image across 5 groups.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import hog


def _load_gray(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to a fixed size for consistent feature extraction
    gray = cv2.resize(gray, (256, 256))
    return gray


# ── 1. SHAPE FEATURES (~15) ──────────────────────────────────────────────────

def shape_features(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / (h + 1e-6)
    else:
        area, perimeter, aspect_ratio = 0.0, 0.0, 1.0
        x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]

    M = cv2.moments(binary)
    hu = cv2.HuMoments(M).flatten()  # 7 values
    # log-transform for numerical stability
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    bbox_area = w * h
    features = np.concatenate([
        hu,                                          # 7
        [area, perimeter, aspect_ratio, bbox_area,   # 4
         w / (gray.shape[1] + 1e-6),                 # 1 normalised width
         h / (gray.shape[0] + 1e-6),                 # 1 normalised height
         area / (bbox_area + 1e-6)],                 # 1 extent / solidity proxy
    ])  # total: 14
    return features.astype(np.float32)


# ── 2. JOINT SPACE WIDTH (~20) ───────────────────────────────────────────────

def joint_space_features(gray):
    h, w = gray.shape
    # Centre vertical strip (middle 40% of width)
    x0 = int(w * 0.30)
    x1 = int(w * 0.70)
    strip = gray[:, x0:x1]

    row_means = strip.mean(axis=1)  # mean intensity per row
    joint_row = int(np.argmax(row_means))

    # Band around the joint row (±10% of height)
    band_h = max(1, int(h * 0.10))
    band = gray[max(0, joint_row - band_h):min(h, joint_row + band_h), x0:x1]

    band_min = float(band.min())
    band_mean = float(band.mean())
    band_std = float(band.std())

    # Width of bright region at 17 evenly-spaced horizontal positions
    threshold = band_mean
    widths = []
    for row_idx in np.linspace(max(0, joint_row - band_h), min(h - 1, joint_row + band_h), 17, dtype=int):
        row = gray[row_idx, x0:x1]
        bright = (row > threshold).sum()
        widths.append(bright)
    widths = np.array(widths, dtype=np.float32)

    features = np.concatenate([
        [band_min, band_mean, band_std,              # 3
         joint_row / (h + 1e-6)],                    # 1 relative position
        widths,                                       # 17
    ])  # total: 21
    return features.astype(np.float32)


# ── 3. TEXTURE FEATURES (~16) ────────────────────────────────────────────────

def texture_features(gray):
    h = gray.shape[0]
    # Sub-region: bottom 30% (tibial plateau trabecular bone)
    sub = gray[int(h * 0.70):, :]

    # GLCM on 8-bit sub-region
    sub8 = (sub // 16).astype(np.uint8)  # quantise to 16 levels
    glcm = graycomatrix(sub8, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_feats = np.array([graycoprops(glcm, p)[0, 0] for p in props], dtype=np.float32)

    # LBP histogram (8 neighbours, radius 1, uniform → 10 bins)
    lbp = local_binary_pattern(sub, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 8), density=True)

    mean_val = float(sub.mean())
    std_val = float(sub.std())

    features = np.concatenate([glcm_feats, lbp_hist, [mean_val, std_val]])  # 6+8+2 = 16
    return features.astype(np.float32)


# ── 4. EDGE / OSTEOPHYTE FEATURES (~25) ──────────────────────────────────────

def edge_features(gray):
    h, w = gray.shape
    top = gray[:int(h * 0.20), :]
    bottom = gray[int(h * 0.80):, :]

    def _canny_counts(region):
        edges = cv2.Canny(region, 50, 150)
        mid = region.shape[1] // 2
        return float(edges[:, :mid].sum() / 255), float(edges[:, mid:].sum() / 255)

    top_med, top_lat = _canny_counts(top)
    bot_med, bot_lat = _canny_counts(bottom)

    # Sobel on full image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mean = float(mag.mean())
    sobel_std = float(mag.std())
    sobel_max = float(mag.max())

    # HOG on full image (9 orientations, 2x2 cells of 8x8 pixels → variable length)
    # Fix output size by using a fixed resize and cells_per_block
    gray_hog = cv2.resize(gray, (128, 128))
    hog_feats = hog(
        gray_hog,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    # Take first 15 values to keep vector compact (total HOG will be > 100)
    hog_summary = hog_feats[:15]

    features = np.concatenate([
        [top_med, top_lat, bot_med, bot_lat,          # 4
         sobel_mean, sobel_std, sobel_max],            # 3
        hog_summary,                                   # 15
    ])  # total: 22
    return features.astype(np.float32)


# ── 5. GLOBAL STATISTICS (~8) ────────────────────────────────────────────────

def global_stats(gray):
    from scipy.stats import skew, kurtosis
    flat = gray.flatten().astype(np.float64)
    half = gray.shape[0] // 2
    top_half = gray[:half, :].flatten().astype(np.float64)
    bot_half = gray[half:, :].flatten().astype(np.float64)
    features = np.array([
        flat.mean(), flat.std(), skew(flat), kurtosis(flat),  # 4
        top_half.mean(), top_half.std(),                       # 2
        bot_half.mean(), bot_half.std(),                       # 2
    ], dtype=np.float32)
    return features  # total: 8


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def extract_features(image_path):
    """Return a 1D float32 feature vector (~81 values) for one image."""
    gray = _load_gray(image_path)
    parts = [
        shape_features(gray),        # ~14
        joint_space_features(gray),  # ~21
        texture_features(gray),      # ~16
        edge_features(gray),         # ~22
        global_stats(gray),          # ~8
    ]
    return np.concatenate(parts)
