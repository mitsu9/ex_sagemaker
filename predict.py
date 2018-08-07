from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.model import TensorFlowPredictor
from sagemaker.tensorflow.predictor import tf_json_deserializer, tf_json_serializer
import tensorflow as tf
import numpy as np

def run():
    # params
    endpoint = "xxxxx" # SageMaker Endpointの名前
    input_length = 185 # このlengthになるように0-paddingする

    # create instance
    predictor = TensorFlowPredictor(endpoint)
    predictor.serializer = tf_json_serializer
    predictor.deserializer = tf_json_deserializer
    predictor.content_type = "application/json"
    predictor.accept = "application/json"

    # prepare data
    data = [[8, 7005, 4568, 4411, 3835, 26, 4771, 13, 1344, 4878, 4, 2601, 3231, 79, 40, 605, 444, 7, 4771, 9, 2739, 21, 40, 2542, 552, 506, 779, 78, 347, 1088, 57, 5, 3293, 711, 14, 1326, 5, 351, 15, 20, 2170, 54, 375, 1133, 57, 104, 441, 9, 1779, 3702, 305, 202, 6338, 90, 4085, 54, 2763, 4, 58, 20, 40, 203, 7, 4771, 15, 7920, 367, 104, 917, 9, 2857, 323, 90, 16, 1725, 4, 831, 3386, 608, 7219, 1072, 9, 2733, 4, 6843, 759, 217, 4, 0]]
    padded_data = [[0] * (input_length - len(d)) + d for d in data]
    tensor_proto = tf.make_tensor_proto(values=np.asarray(padded_data), shape=[len(data), len(padded_data[0])], dtype=tf.float32)

    # debug用 実際に送信しているリクエストのbodyはこれ
    body = predictor.serializer(tensor_proto)
    print(body)

    # predict
    result = predictor.predict(tensor_proto)
    print(result)

if __name__ == '__main__':
    run()
