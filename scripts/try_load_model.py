import tensorflow as tf

MODEL_PATH = 'trained_model.keras'
try:
    m = tf.keras.models.load_model(MODEL_PATH)
    print('Loaded OK, input shape:', m.input_shape)
except Exception as e:
    print('ERROR:', type(e), e)
