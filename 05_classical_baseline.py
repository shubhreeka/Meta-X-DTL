"""Step 5: classical baseline. Hand-crafted features + GA-PSO feature selection + SVM.

Features per image (enhanced with the PSO parameters):
  colour  : mean / std of L, a, b channels (6)
  texture : GLCM contrast, dissimilarity, homogeneity, energy, correlation (5)
  LBP     : uniform LBP histogram, P=24, R=3 (26)
Hybrid GA-PSO searches a continuous vector in [0,1]^37; a feature is selected if > 0.5.
Fitness = 1 - group-aware 5-fold CV balanced accuracy on TRAIN+VAL + 0.01 * fraction of features used.
(Folds never split a photo from its augmented copies.)
Output: results/classical_baseline.json
"""
import json

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import ARTIFACTS, RESULTS, SEED
from src.enhance import enhance
from src.metaheuristics import ga_pso

FEATURE_NAMES = ([f"{c}_{s}" for c in "Lab" for s in ("mean", "std")]
                 + ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
                 + [f"lbp_{i}" for i in range(26)])


def handcrafted(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    colour = np.concatenate([lab.mean(0), lab.std(0)])[[0, 3, 1, 4, 2, 5]]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray // 4, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64, symmetric=True, normed=True)
    texture = [graycoprops(glcm, p).mean() for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")]
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(27), density=True)
    return np.concatenate([colour, texture, hist])


def main():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    p = json.loads((RESULTS / "enhancement.json").read_text())
    images = np.load(ARTIFACTS / "images.npy")
    labels = np.load(ARTIFACTS / "labels.npy")
    split = np.load(ARTIFACTS / "split.npy")
    groups = np.load(ARTIFACTS / "groups.npy")
    X = np.stack([handcrafted(enhance(im, p["alpha"], p["beta"], p["clahe_clip"])) for im in images])
    tr, te = split != "test", split == "test"  # train+val for CV, test held out
    Xtr, ytr, gtr = X[tr], labels[tr], groups[tr]
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr))

    def svm():
        return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True, random_state=SEED))

    def fitness(v):
        mask = v > 0.5
        if mask.sum() == 0:
            return 1.0
        score = cross_val_score(svm(), Xtr[:, mask], ytr, cv=cv, scoring="balanced_accuracy").mean()
        return 1 - score + 0.01 * mask.mean()

    dim = X.shape[1]
    best, cost, hist = ga_pso(fitness, [0] * dim, [1] * dim, pop_size=16, iters=15, seed=SEED)
    mask = best > 0.5

    out = {}
    for name, m in (("all_features", np.ones(dim, bool)), ("ga_pso_selected", mask)):
        clf = svm().fit(Xtr[:, m], ytr)
        prob = clf.predict_proba(X[te][:, m])[:, 1]
        pred = (prob >= 0.5).astype(int)
        out[name] = {"n_features": int(m.sum()),
                     "cv_balanced_accuracy": round(float(cross_val_score(svm(), Xtr[:, m], ytr, cv=cv, scoring="balanced_accuracy").mean()), 4),
                     "test_accuracy": round(float(accuracy_score(labels[te], pred)), 4),
                     "test_balanced_accuracy": round(float(balanced_accuracy_score(labels[te], pred)), 4),
                     "test_f1": round(float(f1_score(labels[te], pred)), 4),
                     "test_roc_auc": round(float(roc_auc_score(labels[te], prob)), 4)}
    out["selected_features"] = [FEATURE_NAMES[i] for i in np.where(mask)[0]]
    out["ga_pso_history"] = [round(h, 4) for h in hist]
    (RESULTS / "classical_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "ga_pso_history"}, indent=2))


if __name__ == "__main__":
    main()
