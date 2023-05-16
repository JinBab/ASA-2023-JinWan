import csv

import numpy as np
import tensorflow as tf

RANDOM_SEED = 42

dataset = 'keypoint.csv'
model_save_path = 'keypoint_classifier.hdf5'
tflite_save_path = 'keypoint_classifier.tflite'

# CHANGE THIS TO THE NUMBER OF CLASSES WE ARE GOING TO TRAIN
NUM_CLASSES = 6

# CHANGE THE usecols parameter TO THE DATASET DIMENSION
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (33 * 3) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))


model = tf.keras.models.Sequential([
    # CHANGE THE INPUT LAYER TO THE COLUMN LENGTH
    tf.keras.layers.Input((33 * 3, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Model checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# Callback for early stopping
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Model compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_dataset,
    y_dataset,
    epochs=1000,
    batch_size=128,
    validation_data=(X_dataset, y_dataset),
    callbacks=[cp_callback, es_callback]
)

# Save as a model dedicated to inference
model.save(model_save_path, include_optimizer=False)

# Transform model (quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)