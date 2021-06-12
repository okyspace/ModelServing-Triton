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
MODEL_NAME = 'reid'
MODEL_VERSIONS = ['1']
INPUT_SHAPE = (384, 128)

# hardcoded images for testing
IMAGES = ['data/0.png', 'data/5.png', 'data/7.png']


def get_embedding(ip, port, model_name, model_version, input_name, input_datatype, output_name, output_datatype, output_shape, images):
    batch_size = len(images)
    input_data = []
    for i in range(batch_size):
        img_path = images[i]
        img = preprocessing(img_path)
        input_data.append(img)

    # adjust data according to model input shape (-1, 3, 384, 128) --> (NCHW)
    input_data = torch.from_numpy(np.array(input_data)).float().permute(0, 3, 2, 1)
    return infer(ip, port, model_name, model_version, input_name, input_datatype, input_data, output_name, output_datatype, output_shape) 


def preprocessing(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SHAPE)
    img = img.astype(np.float)   
    return img


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
