"""
Phase 2B — Landmark-guided hand-crafted features (~81 values), using
baseline/landmarks.py points instead of the blind image-fraction heuristics
in baseline/features.py.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import hog

from baseline.landmarks import detect_landmarks, WORK_SIZE, X_FRACTIONS

N_POS = len(X_FRACTIONS)  # 5 sampled x-positions across the joint width


def _load_gray(image_path, size=WORK_SIZE):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size))


# ── 1. SHAPE FEATURES (~20) — landmark distances & alignment angles ─────────

def _angle(p1, p2):
    """Angle (deg) of the line p1->p2 relative to vertical (0 = straight down)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dx, dy + 1e-6)))


def shape_features(gray, lm):
    h, w = gray.shape
    femur_y = np.array([lm[f"femur_{i}"][1] for i in range(1, N_POS + 1)])
    tibia_y = np.array([lm[f"tibia_{i}"][1] for i in range(1, N_POS + 1)])

    joint_c = lm["joint_3"]
    left, right = lm["bone_left_margin"], lm["bone_right_margin"]
    femur_span = abs(lm["femur_5"][0] - lm["femur_1"][0])
    tibia_span = abs(lm["tibia_5"][0] - lm["tibia_1"][0])
    bone_width = abs(right[0] - left[0])

    varus_valgus_angle = _angle(lm["femur_shaft"], joint_c) - _angle(joint_c, lm["tibia_shaft"])
    femur_axis_angle = _angle(lm["femur_shaft"], lm["femur_3"])
    tibia_axis_angle = _angle(lm["tibia_3"], lm["tibia_shaft"])
    femur_tibia_offset = (lm["femur_3"][0] - lm["tibia_3"][0]) / (w + 1e-6)
    symmetry = (joint_c[0] - left[0]) / (bone_width + 1e-6)
    span_ratio = femur_span / (tibia_span + 1e-6)

    features = np.concatenate([
        (femur_y - femur_y.mean()) / (h + 1e-6),   # 5 — condyle curvature (SSM-like)
        (tibia_y - tibia_y.mean()) / (h + 1e-6),    # 5 — plateau curvature (SSM-like)
        [varus_valgus_angle, femur_axis_angle, tibia_axis_angle,  # 3
         bone_width / (w + 1e-6),                    # 1
         femur_span / (w + 1e-6),                     # 1
         tibia_span / (w + 1e-6),                     # 1
         femur_tibia_offset,                          # 1
         symmetry,                                     # 1
         span_ratio],                                  # 1
    ])  # total: 5+5+9 = 19
    return features.astype(np.float32)


# ── 2. JOINT SPACE WIDTH (~15) — measured at the true landmark-located joint ─

def joint_space_features(gray, lm):
    h, w = gray.shape
    jsw = np.array([
        lm[f"tibia_{i}"][1] - lm[f"femur_{i}"][1] for i in range(1, N_POS + 1)
    ], dtype=np.float64)

    medial = jsw[:2].mean()
    lateral = jsw[3:].mean()
    central = jsw[2]

    joint_row = int(round(np.mean([lm[f"joint_{i}"][1] for i in range(1, N_POS + 1)])))
    x0 = int(lm["bone_left_margin"][0])
    x1 = max(x0 + 1, int(lm["bone_right_margin"][0]))
    band = gray[max(0, joint_row - 2):min(h, joint_row + 3), x0:x1]
    band_mean = float(band.mean()) if band.size else 0.0
    band_min = float(band.min()) if band.size else 0.0

    features = np.concatenate([
        jsw,                                                          # 5
        [medial, lateral, central, jsw.min(), jsw.max(), jsw.std(),   # 6
         medial - lateral, medial / (lateral + 1e-6),                  # 2
         band_mean, band_min],                                         # 2
    ]).astype(np.float32)  # total: 5+6+2+2 = 15
    return features


# ── 3. TEXTURE FEATURES (~16) — tibial plateau ROI from landmarks ───────────

def texture_features(gray, lm):
    h, w = gray.shape
    tibia_y = [lm[f"tibia_{i}"][1] for i in range(1, N_POS + 1)]
    y0 = int(min(tibia_y))
    y1 = min(h, y0 + max(4, int(h * 0.15)))
    x0 = int(lm["bone_left_margin"][0])
    x1 = max(x0 + 1, int(lm["bone_right_margin"][0]))
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        sub = gray[int(h * 0.70):, :]  # degenerate fallback

    sub8 = (sub // 16).astype(np.uint8)
    glcm = graycomatrix(sub8, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_feats = np.array([graycoprops(glcm, p)[0, 0] for p in props], dtype=np.float32)

    lbp = local_binary_pattern(sub, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 8), density=True)

    features = np.concatenate([glcm_feats, lbp_hist, [float(sub.mean()), float(sub.std())]])
    return features.astype(np.float32)  # 6+8+2 = 16


# ── 4. EDGE / OSTEOPHYTE FEATURES (~22) — margins from landmarks ────────────

def edge_features(gray, lm):
    h, w = gray.shape
    femur_y = [lm[f"femur_{i}"][1] for i in range(1, N_POS + 1)]
    tibia_y = [lm[f"tibia_{i}"][1] for i in range(1, N_POS + 1)]
    x0 = int(lm["bone_left_margin"][0])
    x1 = max(x0 + 1, int(lm["bone_right_margin"][0]))
    mid_x = (x0 + x1) // 2 - x0
    pad = max(2, int(h * 0.05))

    top = gray[max(0, int(min(femur_y)) - pad):min(h, int(max(femur_y)) + 2), x0:x1]
    bottom = gray[max(0, int(min(tibia_y)) - 2):min(h, int(max(tibia_y)) + pad), x0:x1]

    def _canny_counts(region):
        if region.size == 0:
            return 0.0, 0.0
        edges = cv2.Canny(region, 50, 150)
        m = max(1, region.shape[1] // 2)
        return float(edges[:, :m].sum() / 255), float(edges[:, m:].sum() / 255)

    top_med, top_lat = _canny_counts(top)
    bot_med, bot_lat = _canny_counts(bottom)

    y0 = max(0, int(min(femur_y)) - pad)
    y1 = min(h, int(max(tibia_y)) + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        roi = gray
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mean, sobel_std, sobel_max = float(mag.mean()), float(mag.std()), float(mag.max())

    roi_hog = cv2.resize(roi, (128, 128))
    hog_feats = hog(roi_hog, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    hog_summary = hog_feats[:15]

    features = np.concatenate([
        [top_med, top_lat, bot_med, bot_lat, sobel_mean, sobel_std, sobel_max],  # 7
        hog_summary,                                                             # 15
    ])
    return features.astype(np.float32)  # 22


# ── 5. GLOBAL STATISTICS (~8) — unchanged ────────────────────────────────────

def global_stats(gray):
    from scipy.stats import skew, kurtosis
    flat = gray.flatten().astype(np.float64)
    half = gray.shape[0] // 2
    top_half = gray[:half, :].flatten().astype(np.float64)
    bot_half = gray[half:, :].flatten().astype(np.float64)
    return np.array([
        flat.mean(), flat.std(), skew(flat), kurtosis(flat),
        top_half.mean(), top_half.std(),
        bot_half.mean(), bot_half.std(),
    ], dtype=np.float32)  # 8


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def extract_landmark_features(image_path):
    """Return a 1D float32 feature vector (~81 values) for one image, guided
    by baseline/landmarks.py's automatically detected landmark points."""
    gray = _load_gray(image_path)
    lm = detect_landmarks(image_path)
    parts = [
        shape_features(gray, lm),        # ~19
        joint_space_features(gray, lm),  # ~15
        texture_features(gray, lm),      # ~16
        edge_features(gray, lm),         # ~22
        global_stats(gray),              # ~8
    ]
    return np.concatenate(parts)
