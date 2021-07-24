#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import dagshub

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = np.float32(x_train[..., np.newaxis] / 255), np.float32(x_test[..., np.newaxis] / 255.0)

model = Sequential([
    # Deliberately bad model architecture for training SPEED
    InputLayer((28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Conv2D(32, kernel_size=4, strides=2, activation='relu'),
    Conv2D(16, kernel_size=4, strides=1, activation='relu'),
    Conv2D(10, kernel_size=3, strides=1, activation='softmax'),
    Flatten()
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

class DAGsHubLogger(Callback):
    def __init__(self):
        self.logger = dagshub.DAGsHubLogger('metrics.csv', should_log_hparams=False)
    def on_epoch_end(self, epoch, logs=None):
        self.logger.log_metrics(logs, step_num=epoch)
    def on_train_end(self, logs=None):
    	self.logger.close()

class BestMetricsLogger(Callback):
    def on_train_end(self, logs=None):
        with open('metrics.yaml', 'w') as m:
            print(f'trainAcc: {logs["accuracy"]}\ntestAcc: {logs["val_accuracy"]}', file=m)

cb_logger = DAGsHubLogger()
cb_checkpoint = ModelCheckpoint('model.h5')
cb_metrics = BestMetricsLogger()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, callbacks=[cb_logger, cb_checkpoint, cb_metrics])
history.history['epochs'] = list(np.array(history.epoch) + 1)
history = history.history

for key in history.keys():
    if key.endswith("accuracy"):
        plt.plot('epochs', key, data=history, label=key)
plt.legend()
plt.show()
