"""Step 6: end-to-end check of the full model on the TEST set + Grad-CAM figure.

Verifies that the assembled inference model (VGG16 + scaler + optimised head) reproduces
the test metrics, then renders Grad-CAM explanations for sample test images.
Outputs: results/gradcam_examples.jpg, results/final_test_metrics.json
"""
import json

import numpy as np
from PIL import Image, ImageDraw

from src.config import ARTIFACTS, CLASSES, RESULTS, SEED
from src.enhance import enhance
from src.gradcam import build_full_model, overlay, predict_with_gradcam


def main():
    from sklearn.metrics import accuracy_score, roc_auc_score

    p = json.loads((RESULTS / "enhancement.json").read_text())
    images = np.load(ARTIFACTS / "images.npy")
    labels = np.load(ARTIFACTS / "labels.npy")
    split = np.load(ARTIFACTS / "split.npy")
    te = np.where(split == "test")[0]

    model = build_full_model()
    probs, cams = [], {}
    for i in te:
        e = enhance(images[i], p["alpha"], p["beta"], p["clahe_clip"])
        prob, cam = predict_with_gradcam(model, e)
        probs.append(prob)
        cams[i] = (e, cam, prob)
    probs = np.array(probs)
    y = labels[te]
    final = {"test_accuracy": round(float(accuracy_score(y, probs >= 0.5)), 4),
             "test_roc_auc": round(float(roc_auc_score(y, probs)), 4), "n_test": int(len(te))}
    (RESULTS / "final_test_metrics.json").write_text(json.dumps(final, indent=2))
    print(final)

    # Figure: 4 correctly classified malignant + 4 benign test images
    rng = np.random.default_rng(SEED)
    pick = []
    for cls in (1, 0):
        ok = [i for i, pr in zip(te, probs) if labels[i] == cls and (pr >= 0.5) == cls]
        pick += list(rng.choice(ok, min(4, len(ok)), replace=False))
    T = 224
    sheet = Image.new("RGB", (len(pick) * T, 2 * T + 22), "white")
    d = ImageDraw.Draw(sheet)
    for k, i in enumerate(pick):
        e, cam, prob = cams[i]
        sheet.paste(Image.fromarray(e), (k * T, 22))
        sheet.paste(Image.fromarray(overlay(e, cam)), (k * T, 22 + T))
        d.text((k * T + 6, 5), f"{CLASSES[labels[i]]}  P(malig)={prob:.2f}", fill="black")
    sheet.save(RESULTS / "gradcam_examples.jpg", quality=90)


if __name__ == "__main__":
    main()
