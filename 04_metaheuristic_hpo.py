"""Step 4: metaheuristic hyper-parameter optimisation of the deep-learning classifier.

PSO, GWO, Firefly and hybrid GA-PSO each search the classification-head hyper-parameters
on top of frozen VGG16 features:
    learning rate (log10), dropout, hidden units, L2 weight decay (log10)
Fitness = validation log-loss (the test set is never used during the search).
Every optimiser's best configuration, plus a hand-picked default, is then retrained and
evaluated once on the held-out TEST set.

Outputs: results/hpo_results.json, results/convergence.png, models/head_best.keras,
         models/feature_scaler.npz
"""
import json
import time

import numpy as np

from src.config import ARTIFACTS, MODELS, RESULTS, SEED
from src.metaheuristics import OPTIMISERS

LOWER = [-4.5, 0.0, 32, -6.0]
UPPER = [-2.0, 0.6, 512, -2.0]
DEFAULT = [-3.0, 0.5, 256, -4.0]  # lr=1e-3, dropout=0.5, 256 units, L2=1e-4
BUDGET = {"PSO": dict(n_particles=8, iters=8), "GWO": dict(n_wolves=8, iters=8),
          "FA": dict(n_fireflies=6, iters=6), "GA-PSO": dict(pop_size=8, iters=8)}


def decode(p):
    return {"lr": float(10 ** p[0]), "dropout": float(p[1]), "units": int(round(p[2])), "l2": float(10 ** p[3])}


def build_head(hp, dim):
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(hp["l2"])
    model = tf.keras.Sequential([
        tf.keras.layers.Input((dim,)),
        tf.keras.layers.Dense(hp["units"], activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(hp["dropout"]),
        tf.keras.layers.Dense(2, activation="softmax", kernel_regularizer=reg),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"]), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_head(hp, data, epochs=60):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    Xtr, ytr, Xva, yva = data
    model = build_head(hp, Xtr.shape[1])
    stop = tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64, verbose=0, callbacks=[stop])
    return model


def metrics(y, prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred), "recall_sensitivity": recall_score(y, pred),
            "specificity": tn / (tn + fp), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    d = np.load(ARTIFACTS / "vgg_features.npz")
    mu, sd = d["X_train"].mean(0), d["X_train"].std(0) + 1e-6
    np.savez(MODELS / "feature_scaler.npz", mean=mu, std=sd)
    Xtr, Xva, Xte = [(d[k] - mu) / sd for k in ("X_train", "X_val", "X_test")]
    ytr, yva, yte = d["y_train"], d["y_val"], d["y_test"]
    data = (Xtr, ytr, Xva, yva)

    results, curves = {}, {}
    for name, opt in OPTIMISERS.items():
        evals = []

        def fitness(p):
            m = train_head(decode(p), data)
            loss = m.evaluate(Xva, yva, verbose=0)[0]
            evals.append(loss)
            return loss

        t0 = time.time()
        best, cost, hist = opt(fitness, LOWER, UPPER, seed=SEED, **BUDGET[name])
        curves[name] = hist
        results[name] = {"best_hparams": decode(best), "val_loss": cost, "evaluations": len(evals),
                         "seconds": round(time.time() - t0, 1), "_vec": list(map(float, best))}
        print(f"{name:7s} val_loss={cost:.4f}  evals={len(evals)}  {decode(best)}")

    results["Default (no optimisation)"] = {"best_hparams": decode(DEFAULT), "_vec": DEFAULT}

    # One-time evaluation on the held-out test set
    best_name, best_loss = None, np.inf
    for name, r in results.items():
        m = train_head(decode(r["_vec"]), data)
        r["val_loss"] = r.get("val_loss", m.evaluate(Xva, yva, verbose=0)[0])
        r["test"] = {k: (round(float(v), 4) if not isinstance(v, list) else v)
                     for k, v in metrics(yte, m.predict(Xte, verbose=0)[:, 1]).items()}
        if name in OPTIMISERS and r["val_loss"] < best_loss:  # model selection on VALIDATION loss
            best_name, best_loss, best_model = name, r["val_loss"], m
        print(f"{name:26s} test acc={r['test']['accuracy']:.4f}  auc={r['test']['roc_auc']:.4f}")
        del r["_vec"]

    best_model.save(MODELS / "head_best.keras")
    results["selected"] = best_name
    (RESULTS / "hpo_results.json").write_text(json.dumps(results, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for name, h in curves.items():
        plt.plot(range(1, len(h) + 1), h, marker="o", label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Best validation log-loss")
    plt.title("Metaheuristic hyper-parameter search: convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "convergence.png", dpi=140)
    print("Selected (lowest validation loss):", best_name)


if __name__ == "__main__":
    main()
