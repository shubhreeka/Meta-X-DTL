"""Step 3: VGG16 transfer-learning features.

Every image is enhanced with the PSO parameters and passed through an ImageNet-pretrained
VGG16 (frozen); block5 feature maps are global-average-pooled to 512-d vectors.
Training images are also embedded horizontally flipped (augmentation, train split only).
Output: data/processed/vgg_features.npz
"""
import json

import numpy as np

from src.config import ARTIFACTS, IMG_SIZE, RESULTS
from src.enhance import enhance


def build_backbone():
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16

    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    return tf.keras.Model(base.input, tf.keras.layers.GlobalAveragePooling2D()(base.output))


def main():
    from tensorflow.keras.applications.vgg16 import preprocess_input

    p = json.loads((RESULTS / "enhancement.json").read_text())
    images = np.load(ARTIFACTS / "images.npy")
    labels = np.load(ARTIFACTS / "labels.npy")
    split = np.load(ARTIFACTS / "split.npy")

    enhanced = np.stack([enhance(im, p["alpha"], p["beta"], p["clahe_clip"]) for im in images])
    net = build_backbone()
    feats = net.predict(preprocess_input(enhanced.astype("float32")), batch_size=32, verbose=1)

    tr = split == "train"
    flipped = net.predict(preprocess_input(enhanced[tr][:, :, ::-1].astype("float32")), batch_size=32, verbose=1)

    np.savez(ARTIFACTS / "vgg_features.npz",
             X_train=np.concatenate([feats[tr], flipped]), y_train=np.concatenate([labels[tr], labels[tr]]),
             X_val=feats[split == "val"], y_val=labels[split == "val"],
             X_test=feats[split == "test"], y_test=labels[split == "test"])
    print("Saved features:", feats.shape, "+", flipped.shape, "flipped train")


if __name__ == "__main__":
    main()
