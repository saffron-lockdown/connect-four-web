import tensorflow as tf

## Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('m1-2.model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
