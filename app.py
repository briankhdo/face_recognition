#!/usr/bin/python3
import re
from io import BytesIO
import decimal
import flask.json
from flask import Flask, send_file, request, jsonify, render_template, send_from_directory, redirect
from PIL import Image, ImageFilter
import requests
import numpy as np
import tensorflow as tf
import os, sys, time
import json

import cv2

from align_dlib import AlignDlib

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1

requests_count = 0
def increase_requests_count():
    global requests_count
    requests_count += 1
    return requests_count

app = Flask(__name__)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_data(nsess, image_data,
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
    result = nsess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

model_file = "./tf_files/output_graph.pb"
label_file = "./tf_files/output_labels.txt"
input_mean = 0
input_std = 255

input_height = 224
input_width = 224

graph = load_graph(model_file)

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

sess = tf.Session(graph=graph)

@app.route('/', methods=["POST"])
def recognize():
    start_time = time.time()
    file = request.files['image']
    nsess = tf.Session()
    if file:
        if increase_requests_count() % 100 == 1:
            nsess.close()
            tf.reset_default_graph()
            nsess = tf.Session()

        data = file.read()
        image = Image.open(BytesIO(data))
        # file_name = '2JoM2I2xtNDsYglO.png'
        # image = Image.open(file_name)
        with BytesIO() as output:
            # image = image.filter(ImageFilter.DETAIL)
            # image = image.filter(ImageFilter.EDGE_ENHANCE)
            # image.save('./test.jpg', format="JPEG")

            from_time = time.time()
            faces, bboxes = detect_faces(image)

            face_time = (time.time() - from_time)
            print("Found %i faces in %ims" % (len(faces), face_time * 1000))

            recognize_results = []

            from_time = time.time()
            for index, face in enumerate(faces):

                image_reader = tf.image.decode_jpeg(
                        face, channels=3, name="jpeg_reader")
                float_caster = tf.cast(image_reader, tf.float32)
                dims_expander = tf.expand_dims(float_caster, 0)
                resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
                normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
                result = nsess.run(normalized)
                
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: result
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
            reg_time = (time.time() - from_time)
            took = time.time() - start_time
            return jsonify({
                    'faces': recognize_results,
                    'took': {
                        'face': float(face_time),
                        'recognition': float(reg_time),
                        'all': float(took)
                    }
                })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

