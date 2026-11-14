"""Step 1: leak-free dataset preparation.

The source dataset mixes original photos with augmented copies (crops, flips,
rotations) of some of them, and the copies cannot be traced by filename. A random
split would put variants of the same photo in both train and test and inflate
accuracy.

This script:
  1. embeds every image with an ImageNet CNN (flip-averaged) to find candidate look-alikes
     (the 20 nearest neighbours of each image),
  2. verifies each candidate pair geometrically with a RANSAC homography on the
     image and its mirror (SIFT keypoints). Enough consistent inliers means it is the same photo, even if
     rotated, cropped, flipped or colour-shifted,
  3. treats each connected component of verified pairs as one "photo group",
  4. splits by group (stratified), so no photo or variant of it appears in two splits.

Outputs data/processed/{images.npy, labels.npy, groups.npy, split.npy, files.json}.
"""
import json

import numpy as np
from PIL import Image, ImageOps
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ARTIFACTS, CLASSES, DATA_ROOT, IMG_SIZE, SEED, SOURCES

K_NEIGHBOURS = 20      # candidate look-alikes per image (by CNN embedding)
MIN_INLIERS = 12       # RANSAC inliers needed to call two images the same photo
SURE_SIMILARITY = 0.86 # embeddings this close are linked without geometric check
                       # (over-merging only makes the split stricter, never leakier)


def keypoint_features(images):
    import cv2

    sift = cv2.SIFT_create(nfeatures=600)
    feats = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        k1, d1 = sift.detectAndCompute(gray, None)
        k2, d2 = sift.detectAndCompute(np.ascontiguousarray(gray[:, ::-1]), None)
        feats.append(((k1, d1), (k2, d2)))
    return feats


def same_photo(fa, fb):
    """Geometric verification: SIFT matches + RANSAC homography (also tries b mirrored)."""
    import cv2

    (ka, da), _ = fa
    best = 0
    for kb, db in fb:  # original and mirrored version of b
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            continue
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            best = max(best, int(mask.sum()))
    return best >= MIN_INLIERS


def load_all():
    files, labels, images = [], [], []
    for label, cls in enumerate(CLASSES):
        for rel in SOURCES[cls]:
            folder = DATA_ROOT / rel
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                try:
                    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                except Exception:
                    continue
                images.append(np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8))
                labels.append(label)
                files.append(f"{rel}/{f.name}")
    return np.stack(images), np.array(labels), files


def embed(images):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

    net = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(images.astype("float32"))
    e = net.predict(x, batch_size=64, verbose=0) + net.predict(x[:, :, ::-1], batch_size=64, verbose=0)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    images, labels, files = load_all()
    print(f"Loaded {len(images)} images  (benign={np.sum(labels == 0)}, malignant={np.sum(labels == 1)})")

    emb = embed(images)
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0)

    feats = keypoint_features(images)
    n = len(images)
    adj = np.zeros((n, n), dtype=bool)
    checked = set()
    for i in range(n):
        for j in np.argsort(-sim[i])[:K_NEIGHBOURS]:
            a, b = min(i, j), max(i, j)
            if (a, b) in checked:
                continue
            checked.add((a, b))
            if sim[a, b] >= SURE_SIMILARITY or same_photo(feats[a], feats[b]):
                adj[a, b] = adj[b, a] = True
    print(f"Checked {len(checked)} candidate pairs, {int(adj.sum() // 2)} confirmed duplicates")
    n_groups, groups = connected_components(adj, directed=False)
    sizes = np.bincount(groups)
    print(f"Near-duplicate groups: {n_groups}  (largest group = {sizes.max()} images, "
          f"{np.sum(sizes > 1)} groups contain duplicates)")

    # A group must not mix labels; if it does, keep the group but report it.
    mixed = [g for g in range(n_groups) if len(set(labels[groups == g])) > 1]
    if mixed:
        print(f"Warning: {len(mixed)} duplicate groups contain both labels")

    # 70 / 15 / 15 split by group, stratified by label.
    split = np.empty(len(images), dtype="<U5")
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(outer.split(images, labels, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    tr, va = next(inner.split(rest_idx, labels[rest_idx], groups[rest_idx]))
    split[rest_idx[tr]], split[rest_idx[va]], split[test_idx] = "train", "val", "test"

    for s in ("train", "val", "test"):
        m = split == s
        print(f"  {s:5s}: {m.sum():5d} images  benign={np.sum(labels[m] == 0):4d}  malignant={np.sum(labels[m] == 1):4d}")
    assert not set(groups[split == "train"]) & set(groups[split == "test"]), "group leak!"

    cross = sim[np.ix_(split == "test", split == "train")].max()
    print(f"Max test-vs-train embedding similarity: {cross:.3f}")
    np.save(ARTIFACTS / "emb.npy", emb)

    np.save(ARTIFACTS / "images.npy", images)
    np.save(ARTIFACTS / "labels.npy", labels)
    np.save(ARTIFACTS / "groups.npy", groups)
    np.save(ARTIFACTS / "split.npy", split)
    (ARTIFACTS / "files.json").write_text(json.dumps(files))
    stats = {
        "n_images": int(len(images)), "n_groups": int(n_groups),
        "k_neighbours": K_NEIGHBOURS, "min_inliers": MIN_INLIERS, "sure_similarity": SURE_SIMILARITY,
        "groups_with_duplicates": int(np.sum(sizes > 1)),
        "split_counts": {s: {"benign": int(np.sum(labels[split == s] == 0)),
                             "malignant": int(np.sum(labels[split == s] == 1))} for s in ("train", "val", "test")},
        "max_test_train_similarity": float(cross),
    }
    (ARTIFACTS.parent.parent / "results" / "data_split.json").write_text(json.dumps(stats, indent=2))
    print("Saved to", ARTIFACTS)


if __name__ == "__main__":
    main()
