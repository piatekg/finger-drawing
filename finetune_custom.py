import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def load_custom_data(data_dir):
    images = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            label_char = filename[0].lower()
            if not label_char.isalpha():
                continue
            label_idx = ord(label_char) - ord('a')

            path = os.path.join(data_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None or img.shape != (28, 28):
                print(f"Skipping {filename} — invalid image.")
                continue

            img = img / 255.0
            img = img.reshape(28, 28, 1)

            images.append(img)
            labels.append(label_idx)

    return np.array(images), np.array(labels)

# Load custom data
X_custom, y_custom = load_custom_data("custom_data")
print(f"Loaded {len(X_custom)} custom samples.")

y_custom_cat = to_categorical(y_custom, num_classes=26)

# Define model architecture (same as your train_emnist.py)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax'),
])

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# Train only on custom data from scratch
model.fit(X_custom, y_custom_cat, epochs=20, batch_size=8, shuffle=True)

# Save model
model.save("custom_letters_scratch.h5")
print("Model trained from scratch on custom data and saved.")
