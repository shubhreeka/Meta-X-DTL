"""Step 7: fine-tune VGG16 (block5) end-to-end with the metaheuristic-selected head.

- Head architecture/regularisation = best configuration found in step 4 (by validation loss).
- Blocks 1-4 frozen, block5 + head trained with a small learning rate.
- Training-only augmentation: flips, rotation, zoom, brightness/contrast jitter.
- Early stopping on VALIDATION loss; the test set is evaluated once at the end,
  with and without flip test-time augmentation (TTA).
The fine-tuned model is kept only if its validation loss beats the frozen-feature model.
Outputs: models/vgg16_finetuned.keras, results/finetune_results.json
"""
import json

import numpy as np

from src.config import ARTIFACTS, IMG_SIZE, MODELS, RESULTS, SEED
from src.enhance import enhance

EPOCHS = 20
BATCH = 32


def main():
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input

    from importlib import import_module
    hpo = import_module("04_metaheuristic_hpo")

    tf.keras.utils.set_random_seed(SEED)
    res = json.loads((RESULTS / "hpo_results.json").read_text())
    hp = res[res["selected"]]["best_hparams"]
    p = json.loads((RESULTS / "enhancement.json").read_text())

    images = np.load(ARTIFACTS / "images.npy")
    labels = np.load(ARTIFACTS / "labels.npy")
    split = np.load(ARTIFACTS / "split.npy")
    X = np.stack([enhance(im, p["alpha"], p["beta"], p["clahe_clip"]) for im in images]).astype("float32")
    Xtr, ytr = X[split == "train"], labels[split == "train"]
    Xva, yva = X[split == "val"], labels[split == "val"]
    Xte, yte = X[split == "test"], labels[split == "test"]

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.15),
    ])
    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(len(Xtr), seed=SEED).batch(BATCH)
                .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    reg = tf.keras.regularizers.l2(hp["l2"])
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(preprocess_input(inp))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(hp["units"], activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    out = tf.keras.layers.Dense(2, activation="softmax", kernel_regularizer=reg)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    stop = tf.keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True)
    hist = model.fit(train_ds, validation_data=(Xva, yva), epochs=EPOCHS, callbacks=[stop], verbose=2)
    val_loss = float(model.evaluate(Xva, yva, verbose=0)[0])

    prob = model.predict(Xte, verbose=0)[:, 1]
    prob_tta = (prob + model.predict(Xte[:, :, ::-1], verbose=0)[:, 1]) / 2
    va_tta = (model.predict(Xva, verbose=0)[:, 1] + model.predict(Xva[:, :, ::-1], verbose=0)[:, 1]) / 2

    out = {
        "head_hparams": hp, "epochs_run": len(hist.history["loss"]),
        "val_loss_finetuned": val_loss, "val_loss_frozen": res[res["selected"]]["val_loss"],
        "val_accuracy_tta": float(np.mean((va_tta >= 0.5) == yva)),
        "test": {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in hpo.metrics(yte, prob).items()},
        "test_tta": {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in hpo.metrics(yte, prob_tta).items()},
        "history": {k: [round(float(v), 4) for v in vs] for k, vs in hist.history.items()},
    }
    out["kept"] = val_loss < out["val_loss_frozen"]
    model.save(MODELS / "vgg16_finetuned.keras")
    (RESULTS / "finetune_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("val_loss_finetuned", "val_loss_frozen", "kept", "test", "test_tta")}, indent=2))


if __name__ == "__main__":
    main()
