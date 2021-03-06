"""
Code to convert tensorflow model to tflite version for running on server
"""
import tensorflow as tf

## Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('models/m6by7-dense-cont-v-new-q-v-itself-40-iters.model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('models/model-40-iters.tflite', 'wb') as f:
  f.write(tflite_model)
