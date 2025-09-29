import tensorflow as tf
import numpy as np
from PIL import Image
import os

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, 'trained_model.keras')
TEST_DIR = os.path.join(BASE, 'test')

files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# helper to infer label (same as before)
def infer_label_from_filename(fname):
    name = fname.lower()
    if 'applecedarrust' in name or 'applecedar' in name:
        return 'Apple___Cedar_apple_rust'
    if 'applescab' in name or 'applescab' in name:
        return 'Apple___Apple_scab'
    if 'corncommonrust' in name:
        return 'Corn_(maize)___Common_rust_'
    if 'potatoearlyblight' in name:
        return 'Potato___Early_blight'
    if 'potatohealthy' in name:
        return 'Potato___healthy'
    if 'tomatoearlyblight' in name:
        return 'Tomato___Early_blight'
    if 'tomatohealthy' in name:
        return 'Tomato___healthy'
    if 'tomatoyellow' in name or 'yellowcurl' in name:
        return 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    return None

model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded.')

modes = {
    '0-1': lambda x: x/255.0,
    '0-255': lambda x: x,
    '-1-1': lambda x: (x/127.5)-1
}

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

for name, fn in modes.items():
    n=0
    top1=0
    top3=0
    for f in files:
        true = infer_label_from_filename(f)
        if true is None:
            continue
        n+=1
        img = Image.open(os.path.join(TEST_DIR,f)).resize((128,128))
        arr = np.array(img).astype(np.float32)
        arr = fn(arr)
        arr = np.expand_dims(arr,0)
        preds = model.predict(arr)[0]
        top3_idx = preds.argsort()[-3:][::-1]
        pred1 = top3_idx[0]
        if class_names[pred1]==true:
            top1+=1
        if true in [class_names[i] for i in top3_idx]:
            top3+=1
    print(f"Mode {name}: Top1 {top1}/{n} = {top1/n:.3f}, Top3 {top3}/{n} = {top3/n:.3f}")
