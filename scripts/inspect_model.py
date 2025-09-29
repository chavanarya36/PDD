import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_model.keras')
print('Loading model from', MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print('Model summary:')
model.summary()
try:
    input_shape = model.input_shape
    print('Model input_shape:', input_shape)
except Exception as e:
    print('Could not get input_shape:', e)

# Print final activation if available
try:
    last = model.layers[-1]
    print('Last layer:', last.name, type(last), getattr(last, 'activation', None))
except Exception as e:
    print('Could not inspect last layer:', e)
