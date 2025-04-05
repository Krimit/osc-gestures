# quantize tflite
import tensorflow as tf
import numpy as np
import pandas as pd

# Load your pre-trained Keras model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Load your dataset
csv_file_path = 'hand_gesture_data.csv'
data = pd.read_csv(csv_file_path)
X = data.drop('label', axis=1).values

# Representative dataset generator
def representative_data_gen():
    for i in range(100):
        input_data = X[i].reshape(1, -1).astype(np.float32)
        yield [input_data]

# Convert the model to a TFLite model with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert and save the quantized model
tflite_model_quant = converter.convert()
with open('hand_gesture_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model_quant)

print("Model quantization complete and saved as 'hand_gesture_model_quantized.tflite'")