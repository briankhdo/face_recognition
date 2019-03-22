#!/usr/bin/python3
import re
from io import BytesIO
import decimal
import flask.json
from flask import Flask, send_file, request, jsonify, render_template, send_from_directory, redirect
from PIL import Image
from flask_basicauth import BasicAuth
import requests
import numpy as np
import tensorflow as tf
import os, sys, time
import json

import cv2

from align_dlib import AlignDlib

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1

app = Flask(__name__)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

model_file = "/root/face_recognition/tf_files/output_graph.pb"
label_file = "/root/face_recognition/tf_files/output_labels.txt"
input_mean = 0
input_std = 255

input_height = 224
input_width = 224

graph = load_graph(model_file)
detection_sess = None
with graph.as_default():
    with tf.Session(config=config,graph=graph) as sess:
        detection_sess = sess

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    result = detection_sess.run(normalized)

    return result

def read_tensor_from_image_data(image_data, 
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    image_reader = tf.image.decode_jpeg(
            image_data, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    result = detection_sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

input_name = "import/Placeholder"
output_name = "import/final_result"
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

def detect_faces(image):
    crop_dim = input_height
    open_cv_image = np.array(image)
    '''Plots the object detection result for a given image.'''
    bbes = align_dlib.getFaceBoundingBoxes(open_cv_image)

    aligned_images = []
    if bbes is not None:
        for bb in bbes:
            aligned = align_dlib.align(crop_dim, open_cv_image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            if aligned is not None:
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                image_data = cv2.imencode('.jpg', aligned)[1].tostring()
                aligned_images.append(image_data)
    return aligned_images, bbes

print("Application loaded")

@app.route('/', methods=["POST"])
def recognize():
    start_time = time.time()
    file = request.files['image']
    if file:
        data = file.read()
        image = Image.open(BytesIO(data))
        # file_name = '2JoM2I2xtNDsYglO.png'
        # image = Image.open(file_name)
        with BytesIO() as output:
            image.save(output, format="JPEG")
            contents = output.getvalue()

            faces, bboxes = detect_faces(image)

            print("Found %i faces" % len(faces))

            recognize_results = []

            for index, face in enumerate(faces):
                image = read_tensor_from_image_data(
                    face,
                    input_height=input_height,
                    input_width=input_width,
                    input_mean=input_mean,
                    input_std=input_std)
                results = detection_sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: image
                })
                results = np.squeeze(results)
                top_k = results.argsort()[-1:][::-1]
                labels = load_labels(label_file)
                classify_result = {}
                for i in top_k:
                   classify_result[labels[i]] = float(results[i])
                box = bboxes[index]
                recognize_results.append({
                        "box": [box.left(), box.top(), box.right(), box.bottom() ],
                        "face": classify_result
                    })
            took = time.time() - start_time
            return jsonify({
                    'faces': recognize_results,
                    'took': float(took)
                })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

