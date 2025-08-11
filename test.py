# test.py
import argparse
import os
import tensorflow as tf
import numpy as np
from model import model
from data_loader_ import test_ds

IMG_HEIGHT = 299
IMG_WIDTH = 299

def load_and_preprocess_image(image_path):
    """Load and preprocess image for model prediction."""
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalize if your model expects it
    return img_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model or predict on a single image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="trained_model.keras",
        help="Path to model weights file."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image for prediction."
    )
    args = parser.parse_args()

    # Load weights
    if os.path.exists(args.weights):
        print(f"[INFO] Loading weights from {args.weights}")
        model.load_weights(args.weights)
    else:
        raise FileNotFoundError(f"[ERROR] No weights found at {args.weights}")

    if args.image:
        # Predict on single image
        print(f"[INFO] Predicting on image: {args.image}")
        img_array = load_and_preprocess_image(args.image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class index: {predicted_class}")
        print(f"Raw prediction scores: {prediction}")
    else:
        # Evaluate on test dataset
        print("[INFO] Evaluating model...")
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")


# python test.py  #Evaluate on test dataset: only when data available
# python test.py --image path/to/your/image.jpg #Predict on a single image:
# python test.py --weights my_model.keras --image leaf.jpg Use custom weights:
