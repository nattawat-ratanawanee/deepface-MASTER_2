import time
import glob
import flask
import io
import os
import base64

import numpy as np
import cv2
import tensorflow as tf

from xml.dom import minidom
from PIL import Image
from convert_utils import *
from deepface_engine import DeepFace

global session_conf
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

global graph

global _deepface
global _faceLiveness

FACE_TMP = 'tmp_face.jpg'
CARD_TMP = 'tmp_idcard.jpg'

@app.route("/image-to-base64string", methods=["POST"])
def image_to_base64string():
    data = {"Error": "", "Status": False}
    http_code = 500
    if flask.request.method == "POST":
        if flask.request.files.get("Image"):
            try:
                image = byteimage_to_string(flask.request.files["Image"].read())

                data["Status"] = True
                data["Image"] = image
                http_code = 200
            except:
                data["Error"] = "Internal Error"
                return flask.jsonify(data), http_code
        else:
            data["Error"] = "Input mismatch"
            http_code = 400
    else:
        data["Error"] = "Method mismatch"
        http_code = 405

    print("Status:", data["Status"])
    return flask.jsonify(data), http_code

@app.route("/detect-face", methods=["POST"])
def detect_face():
    data = {"Error": "", "Status": False}
    http_code = 500
    if flask.request.method == "POST":
        if flask.request.get_json(force=True):
            try:
                input_data = flask.request.get_json(force=True)
                image = string_to_image(input_data["Image"])

                with graph.as_default():
                    face_data = _deepface.extract_face(image)
                
                data["Status"] = face_data["Status"]
                data["Face"] = image_to_string(face_data["Face"])
                data["X"] = face_data["X"]
                data["Y"] = face_data["Y"]
                data["Width"] = face_data["Width"]
                data["Height"] = face_data["Height"]
                http_code = 200
            except:
                data["Error"] = "Internal Error"
                return flask.jsonify(data), http_code
        else:
            data["Error"] = "Input mismatch"
            http_code = 400
    else:
        data["Error"] = "Method mismatch"
        http_code = 405

    print("Status:", data["Status"])
    return flask.jsonify(data), http_code

@app.route("/generate-template", methods=["POST"])
def generate_template():
    data = {"Error": "", "Status": False}
    http_code = 500
    if flask.request.method == "POST":
        if flask.request.get_json(force=True):
            try:
                input_data = flask.request.get_json(force=True)
                face = string_to_image(input_data["Face"])

                Image.fromarray(face).save(FACE_TMP)

                list_img = [FACE_TMP]
                with graph.as_default():
                    extracted_template = _deepface.get_embeddings(list_img)
                
                os.remove(FACE_TMP)

                data["Template"] = numpy_to_Base64_to_string(extracted_template[0].astype(np.float32))
                data["Status"] = True
                http_code = 200
            except:
                data["Error"] = "Internal Error"
                return flask.jsonify(data), http_code
        else:
            data["Error"] = "Input mismatch"
            http_code = 400
    else:
        data["Error"] = "Method mismatch"
        http_code = 405

    print("Status:", data["Status"])
    return flask.jsonify(data), http_code

@app.route("/match-template", methods=["POST"])
def match_template():
    data = {"Error": "", "Status": False}
    http_code = 500
    if flask.request.method == "POST":
        if flask.request.get_json(force=True):
            try:
                input_data = flask.request.get_json(force=True)
                thresh = float(input_data['Threshold'])
                template1 = input_data["Template1"]
                template2 = input_data["Template2"]
                
                template1_array = string_to_Base64_to_numpy(template1)
                template2_array = string_to_Base64_to_numpy(template2)

                match_result = _deepface.is_match(template1_array, template2_array, thresh)

                data["Status"] = True
                data["Matched"] = match_result[0]
                data["Score"] = match_result[1]
                http_code = 200
            except:
                data["Error"] = "Internal Error"
                return flask.jsonify(data), http_code
        else:
            data["Error"] = "Input mismatch"
            http_code = 400
    else:
        data["Error"] = "Method mismatch"
        http_code = 405

    print("Status:", data["Status"], " Matched:", data["Matched"], " Score:", data["Score"])
    return flask.jsonify(data), http_code

@app.route("/check-face-liveness", methods=["POST"])
def check_face_liveness():
    data = {"Error": "", "Status": False}
    http_code = 500
    if flask.request.method == "POST":
        if flask.request.files.get("Image"):
            try:
                image = byteimage_to_image(flask.request.files["Image"].read())
                with graph.as_default():
                    score, predict = _faceLiveness.process(image, 0)

                if score == -999:
                    data["Error"] = "No Face Detected"
                    http_code = 404
                else:
                    data["Status"] = True
                    data["Score"] = float(score)
                    data["Prediction"] = predict
                    http_code = 200
                    print(" Prediction:", data["Prediction"], " Score:", data["Score"])
            except:
                data["Error"] = "Internal Error"
                return flask.jsonify(data), http_code
        else:
            data["Error"] = "Input mismatch"
            http_code = 400
    else:
        data["Error"] = "Method mismatch"
        http_code = 405
    print("Status:", data["Status"])
    return flask.jsonify(data), http_code

@app.route("/check-status", methods=["GET"])
def check_status():
    data = {'status':True}
    print(data)
    return flask.jsonify(data)

if __name__ == "__main__":
    graph = tf.get_default_graph()

    conf = minidom.parse("./config.xml")
    address = conf.getElementsByTagName('address')[0].firstChild.data
    port = conf.getElementsByTagName('port')[0].firstChild.data
    gpu = eval(conf.getElementsByTagName('gpu')[0].firstChild.data)
    cpu_core = int(conf.getElementsByTagName('cpu_core')[0].firstChild.data)

    model_dir = conf.getElementsByTagName('model_dir')[0].firstChild.data
    facedetection_model_path = model_dir + conf.getElementsByTagName('facedetection')[0].getElementsByTagName('model_path')[0].firstChild.data

    liveness_type = conf.getElementsByTagName('liveness_type')[0].firstChild.data
    liveness_threshold = float(conf.getElementsByTagName('liveness_'+liveness_type)[0].getElementsByTagName('threshold')[0].firstChild.data)
    liveness_model_path = model_dir + conf.getElementsByTagName('liveness_'+liveness_type)[0].getElementsByTagName('model_path')[0].firstChild.data
    liveness_image_size = int(conf.getElementsByTagName('liveness_'+liveness_type)[0].getElementsByTagName('image_size')[0].firstChild.data)

    deepface_model = conf.getElementsByTagName('deepface')[0].getElementsByTagName('model')[0].firstChild.data
    deepface_vgg_model_path = model_dir + conf.getElementsByTagName('deepface')[0].getElementsByTagName('vgg_model_path')[0].firstChild.data
    deepface_mtcnn_model_path = model_dir + conf.getElementsByTagName('deepface')[0].getElementsByTagName('mtcnn_model_path')[0].firstChild.data
    deepface_pooling = conf.getElementsByTagName('deepface')[0].getElementsByTagName('pooling')[0].firstChild.data
    deepface_image_size = int(conf.getElementsByTagName('deepface')[0].getElementsByTagName('image_size')[0].firstChild.data)

    if not gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=cpu_core,
            inter_op_parallelism_threads=cpu_core)
        sess = tf.Session(config=session_conf)

    if liveness_type == "full":
        from liveness_detection_full import LivenessDetection
    elif liveness_type == "mini":
        from liveness_detection_mini import LivenessDetection

    _deepface = DeepFace(deepface_model, deepface_pooling, deepface_image_size, deepface_mtcnn_model_path, deepface_vgg_model_path)
    _faceLiveness = LivenessDetection(facedetection_model_path, liveness_model_path, liveness_threshold, liveness_image_size)
    _faceLiveness.initial()
    app.run(host=address, port=port, debug=False)
