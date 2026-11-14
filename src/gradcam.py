"""Full inference model (VGG16 + trained head) and Grad-CAM (Selvaraju et al., 2017)."""
import cv2
import numpy as np

from src.config import IMG_SIZE, MODELS

LAST_CONV = "block5_conv3"


def build_full_model():
    """VGG16 conv base -> GAP -> standardise (train statistics) -> optimised head."""
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16

    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    head = tf.keras.models.load_model(MODELS / "head_best.keras")
    s = np.load(MODELS / "feature_scaler.npz")
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = (x - s["mean"].astype("float32")) / s["std"].astype("float32")
    out = head(x)
    return tf.keras.Model(base.input, [base.get_layer(LAST_CONV).output, out])


def predict_with_gradcam(model, enhanced_rgb):
    """Returns (P(malignant), heatmap in [0,1] at IMG_SIZE) for one enhanced RGB uint8 image."""
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import preprocess_input

    x = preprocess_input(enhanced_rgb.astype("float32")[None])
    with tf.GradientTape() as tape:
        conv, probs = model(x, training=False)
        cls = tf.argmax(probs[0])
        score = probs[:, cls]
    grads = tape.gradient(score, conv)[0]            # d(score)/d(feature maps)
    weights = tf.reduce_mean(grads, axis=(0, 1))     # global-average-pooled gradients
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam / (cam.max() + 1e-8)
    return float(probs[0, 1]), cam


def overlay(rgb, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8((1 - alpha) * rgb + alpha * heat)
