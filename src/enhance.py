"""Image enhancement used before the CNN (parameters found by PSO)."""
import cv2
import numpy as np


def enhance(img_rgb, alpha, beta, clip):
    """Contrast/brightness (alpha, beta) + CLAHE on the L channel (clip limit)."""
    out = cv2.convertScaleAbs(img_rgb, alpha=float(alpha), beta=float(beta))
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhancement_quality(img_rgb):
    """Classic metaheuristic image-enhancement objective (Munteanu & Rosa style):
    reward information content (entropy) and edge strength, penalise clipping."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    nz = hist[hist > 0]
    entropy = -np.sum(nz * np.log2(nz))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edges = np.mean(np.hypot(gx, gy)) / 255.0
    clipped = hist[0] + hist[255]  # fraction of saturated pixels
    return entropy * np.log1p(10 * edges) * (1 - clipped) ** 4
