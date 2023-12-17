import grpc
import numpy as np
import tensorflow_serving_api
import os

from io import BytesIO
from urllib import request as url_request
from PIL import Image

# from tensorflow_serving_api import predict_pb2
# from tensorflow_serving_api import predict_service_pb2_grpc

from flask import Flask
from flask import request
from flask import jsonify

from gateway.proto import np_tp_protobuf


host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

#create grpc channel to communicate gateway container with tf service container
channel = grpc.insecure_channel(host)
stub = tensorflow_serving_api.predict_service_pb2_grpc.PredictionServiceStub(channel)

#classes
classes = ["Horse", "Human"]

#default img url
img_url = 'https://upload.wikimedia.org/wikipedia/commons/b/be/Domestic_horse_at_Suryachaur_and_the_mountains_in_the_back1.jpg'

#function for download the img
def download_image(url):
    with url_request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

#function for prepare the image
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

#function to prepare the input
def prepare_input(x):
    return x / 255.0

app = Flask('gateway')
@app.route('/predict', methods=['POST'])

def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)

def prepare_response(pb_response):
    ###FILL WITH OUTPUT     
    preds = pb_response.outputs['dense_35'].float_val
    return dict(zip(classes, preds))

def prepare_request(X):
    pb_request = tensorflow_serving_api.predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'hourse_vs_humans'
    pb_request.model_spec.signature_name = 'serving_defauls'
    ###FILL WITH INPUT    
    pb_request.inputs['input_17'].CopyFrom(np_tp_protobuf(X))
    return pb_request

def predict(url=img_url):
    img = download_image(url)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)  
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response

if __name__ == '__main__':
    # url = img_url
    # response = predict(url)
    # print(response)
    app.run(debug = True, host = '0.0.0.0', port=9696)


