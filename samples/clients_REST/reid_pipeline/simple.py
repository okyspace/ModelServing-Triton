import sys
import os
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

import torch
import cv2
import numpy as np

from util import get_server_health
from util import get_model_metadata
from util import infer
from util import parse_server_response


# server and model info
TRITON_IP = '0.0.0.0'
TRITON_HTTP_PORT = '8000'
MODEL_NAME = 'reid_pipeline'
MODEL_VERSIONS = ['1']

# hardcoded images for testing
IMAGES = ['data/7.png', 'data/7.png']


def get_embedding(ip, port, model_name, model_version, input_name, input_datatype, output_name, output_datatype, output_shape, images):
    batch_size = len(images)
    input_data = []
    for i in range(batch_size):
        img_path = images[i]
        img = cv2.imread(img_path)
        input_data.append(img)

    return infer(ip, port, model_name, model_version, input_name, input_datatype, np.array(input_data), output_name, output_datatype, output_shape) 


def postprocessing(response, batch_size):
    shape, embeddings = parse_server_response(response)
    embeddings = np.array(embeddings).reshape(shape)
    for i in range(batch_size):
        print('image #{} embeddings shape: {}'.format(i+1, embeddings[i].shape))


def handle_error(response):
    print("log your error / handle it...")
    print(response)


if __name__ == "__main__":
    # prints server health
    get_server_health(TRITON_IP, TRITON_HTTP_PORT)
    
    # get class from various model versions
    for v in MODEL_VERSIONS:
        print("Model {}, Version: {}".format(MODEL_NAME, v))
        model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape = get_model_metadata(TRITON_IP, TRITON_HTTP_PORT, MODEL_NAME, v)
        server_response = get_embedding(TRITON_IP, TRITON_HTTP_PORT, MODEL_NAME, v, input_name, input_datatype, output_name, output_datatype, output_shape, IMAGES)
        batch_size = len(IMAGES)
        if server_response.status_code == 200:
            print(".... inference SUCCESS ....")
            postprocessing(server_response, batch_size)
        else:
            print('.... inference FAILED ....')
            handle_error(server_response)

        print()
