#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.float32(x_train[..., np.newaxis] / 255), np.float32(x_test[..., np.newaxis] / 255.0)

model = tf.keras.models.Sequential([
  # Deliberately bad model architecture for training SPEED
  tf.keras.layers.InputLayer((28, 28, 1)),
  tf.keras.layers.MaxPooling2D(pool_size=2),
  tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu'),
  tf.keras.layers.Conv2D(16, kernel_size=4, strides=1, activation='relu'),
  tf.keras.layers.Conv2D(10, kernel_size=3, strides=1, activation='softmax'),
  tf.keras.layers.Flatten()
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
history.history['epochs'] = list(np.array(history.epoch) + 1)
history = history.history

import matplotlib.pyplot as plt
for key in history.keys():
    if key.endswith("accuracy"):
        plt.plot('epochs', key, data=history, label=key)
plt.legend()
plt.show()

