'''
RESTFUL APIs: https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md
This python file includes examples to do RESTFUL calls to Triton Server for
1. server health
2. model metadata
3. inference

Alternatively, You can also use Triton Client Libraries to do REST/GRPC calls to Triton Server. Library available in C++ / Python.
'''

import requests
import json


def get_server_health(ip, port):
    # /v2/health/ready
    url = 'http://' + ip + ':' + port + '/v2/health/ready'
    resp = requests.get(url)
    if resp.status_code == 200:
        print(".... server is READY ....\n")
    else:
        print(".... server NOT READY ....\n")


def get_model_metadata(ip, port, model_name, model_version):
    # /v2/models/{MODEL_NAME}/versions/{VERSION}
    url = 'http://' + ip + ':' + port + '/v2/models/' + model_name + '/versions/' + model_version
    resp = requests.get(url)
    if resp.status_code == 200:
        print(".... model metadata is fetched SUCCESS ....")
        resp = resp.json()
        model_name = resp['name']
        model_ver = resp['versions'][0]
        ip = resp['inputs'][0]
        op = resp['outputs'][0]
        input_name = ip['name']
        input_datatype = ip['datatype']
        input_shape = ip['shape']
        output_name = op['name']
        output_datatype = op['datatype']
        output_shape = op['shape']

        print("model_name: {}, model_ver: {}, input:{}, {}, {}, output: {}, {}, {}\n"
            .format(model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape))

        return model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape
    else:
        print("error getting model metadata ...\n")


def infer(ip, port, model_name, model_version, input_name, input_datatype, input_data, output_name, output_datatype, output_shape):
    # /v2/models/{MODEL_NAME}/versions/{VERSION}/infer
    url = 'http://' + ip + ':' + port + '/v2/models/' + model_name + '/versions/' + model_version + '/infer'

    # construct inference request body
    infer_req = {
        "id": "optional-use-if-need-to-check-against-http-return",
        "inputs": [
            {
                "name": input_name,
                "shape": input_data.shape,
                "datatype": input_datatype,
                "data": input_data.tolist()
            }
        ],
        "outputs": [
            {
                "name": output_name,
            }
        ]
    }

    headers = {'Content-type': 'application/json'}
    return requests.post(url, data=json.dumps(infer_req), headers=headers)


def parse_server_response(response):
    r = response.json()
    output = r['outputs'][0]
    shape = output['shape']
    data = output['data']
    return shape, data
