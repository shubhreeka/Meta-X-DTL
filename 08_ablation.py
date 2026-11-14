"""Step 8: ablation. Backbone x preprocessing, using the metaheuristic-tuned heads.

Backbones   : VGG16 (paper), ResNet50 (ImageNet, frozen, GAP features)
              (EfficientNetB0 was dropped: its ImageNet weights fail to load in Keras 3.x)
Preprocess  : original photo vs PSO-enhanced photo
Heads       : the best hyper-parameters found by PSO, GWO, FA, GA-PSO (step 4) + default
Ensembles   : average of the four metaheuristic-tuned heads (per backbone x input)
Selection   : the configuration with the lowest VALIDATION log-loss (no regularisation term).
The test set is reported for every row for transparency, but the selected model is the
one decided by validation only.
Outputs: results/ablation.json, results/ablation.csv
"""
import json

import numpy as np

from src.config import ARTIFACTS, IMG_SIZE, RESULTS, SEED
from src.enhance import enhance

BACKBONES = ["VGG16", "ResNet50"]


def extract(name, images):
    import tensorflow as tf
    from tensorflow.keras import applications as A

    net_cls, prep = {"VGG16": (A.VGG16, A.vgg16.preprocess_input),
                     "ResNet50": (A.ResNet50, A.resnet50.preprocess_input),
                     "EfficientNetB0": (A.EfficientNetB0, A.efficientnet.preprocess_input)}[name]
    net = net_cls(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    return net.predict(prep(images.astype("float32")), batch_size=32, verbose=0)


def main():
    from sklearn.metrics import log_loss, roc_auc_score

    from importlib import import_module
    hpo = import_module("04_metaheuristic_hpo")

    images = np.load(ARTIFACTS / "images.npy")
    labels = np.load(ARTIFACTS / "labels.npy")
    split = np.load(ARTIFACTS / "split.npy")
    enh_p = json.loads((RESULTS / "enhancement.json").read_text())
    res = json.loads((RESULTS / "hpo_results.json").read_text())
    heads = {k: v["best_hparams"] for k, v in res.items() if isinstance(v, dict) and "best_hparams" in v}

    variants = {"original": images,
                "pso_enhanced": np.stack([enhance(im, enh_p["alpha"], enh_p["beta"], enh_p["clahe_clip"]) for im in images])}
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    rows = []
    ens_store = {}
    for bb in BACKBONES:
        for vname, imgs in variants.items():
            cache = ARTIFACTS / f"feat_{bb}_{vname}.npz"
            if cache.exists():
                f = np.load(cache)
                F, Fflip = f["F"], f["Fflip"]
            else:
                F = extract(bb, imgs)
                Fflip = extract(bb, imgs[tr][:, :, ::-1])
                np.savez(cache, F=F, Fflip=Fflip)
            Xtr = np.concatenate([F[tr], Fflip]); ytr = np.concatenate([labels[tr], labels[tr]])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr, Xva, Xte = (Xtr - mu) / sd, (F[va] - mu) / sd, (F[te] - mu) / sd
            for hname, hp in heads.items():
                m = hpo.train_head(hp, (Xtr, ytr, Xva, labels[va]))
                pva = m.predict(Xva, verbose=0)[:, 1]
                pte = m.predict(Xte, verbose=0)[:, 1]
                if hname in ("PSO", "GWO", "FA", "GA-PSO"):
                    ens_store.setdefault((bb, vname), []).append((pva, pte))
                row = {"backbone": bb, "input": vname, "head": hname,
                       "val_logloss": round(float(log_loss(labels[va], pva)), 4),
                       "val_auc": round(float(roc_auc_score(labels[va], pva)), 4),
                       "val_accuracy": round(float(np.mean((pva >= 0.5) == labels[va])), 4)}
                row.update({f"test_{k}": (round(float(v), 4) if not isinstance(v, list) else v)
                            for k, v in hpo.metrics(labels[te], pte).items()})
                rows.append(row)
                print(f"{bb:15s} {vname:13s} {hname:26s} val_ll={row['val_logloss']:.3f} "
                      f"val_auc={row['val_auc']:.3f} | test_acc={row['test_accuracy']:.3f} test_auc={row['test_roc_auc']:.3f}")

    for (bb, vname), preds in ens_store.items():
        pva = np.mean([a for a, _ in preds], axis=0)
        pte = np.mean([b for _, b in preds], axis=0)
        row = {"backbone": bb, "input": vname, "head": "Ensemble (PSO+GWO+FA+GA-PSO)",
               "val_logloss": round(float(log_loss(labels[va], pva)), 4),
               "val_auc": round(float(roc_auc_score(labels[va], pva)), 4),
               "val_accuracy": round(float(np.mean((pva >= 0.5) == labels[va])), 4)}
        row.update({f"test_{k}": (round(float(v), 4) if not isinstance(v, list) else v)
                    for k, v in hpo.metrics(labels[te], pte).items()})
        rows.append(row)
        print(f"{bb:15s} {vname:13s} {row['head']:26s} val_ll={row['val_logloss']:.3f} "
              f"val_auc={row['val_auc']:.3f} | test_acc={row['test_accuracy']:.3f} test_auc={row['test_roc_auc']:.3f}")

    best = min(rows, key=lambda r: r["val_logloss"])
    (RESULTS / "ablation.json").write_text(json.dumps({"selected_by_validation": best, "rows": rows}, indent=2))
    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items() if k != "test_confusion_matrix"} for r in rows]).to_csv(
        RESULTS / "ablation.csv", index=False)
    print("\nSELECTED (lowest validation log-loss):", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
