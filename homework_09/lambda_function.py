import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request as request

preprocessor = create_preprocessor('xception', target_size=(200, 200))
interpreter = tflite.Interpreter(model_path='hairstyle-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def preprocess_input(image_url):
    image = preprocessor.from_url(image_url)
    image = np.expand_dims(image, axis=0) 
    return image

def predict(url):
    X = preprocess_input(url)  
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds.tolist()  

def lambda_handler(event, context):
    url = event.get('url')
    if not url:
        return {"error": "Missing 'url' in the request"}
    try:
        result = predict(url)
        return {"predictions": result}
    except Exception as e:
        return {"error": str(e)}
