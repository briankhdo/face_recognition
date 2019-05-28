#!/usr/bin/python3
from io import BytesIO
import glob
import decimal
from PIL import Image, ImageFilter
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

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

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
  result = normalize_sess.run(normalized)

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
  result = normalize_sess.run(normalized)

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

input_height = 128
input_width = 128

graph = load_graph(model_file)

input_name = "import/Placeholder"
output_name = "import/final_result"
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

def detect_face(image):
  crop_dim = input_height
  open_cv_image = np.array(image)
  '''Plots the object detection result for a given image.'''
  bb = align_dlib.getLargestFaceBoundingBox(open_cv_image)

  image_data = None
  if bb is not None:
    aligned = align_dlib.align(crop_dim, open_cv_image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
      aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
      image_data = cv2.imencode('.jpg', aligned)[1].tostring()
  return image_data, bb

sess = tf.Session(graph=graph,config=config)


def recognize(file_name, nsess):
  start_time = time.time()
  image = Image.open(file_name)
  size = 308,231
  image.thumbnail(size)
  # image = image.filter(ImageFilter.DETAIL)
  # image = image.filter(ImageFilter.EDGE_ENHANCE)
  # image.save('./test.jpg', format="JPEG")

  from_time = time.time()
  face, bbox = detect_face(image)

  if face is not None:

    face_time = (time.time() - from_time)

    recognize_results = []

    from_time = time.time()

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
    box = bbox
    recognize_results.append({
        "box": [box.left(), box.top(), box.right(), box.bottom() ],
        "face": classify_result
      })

    reg_time = (time.time() - from_time)
    took = time.time() - start_time
    return {
        'faces': recognize_results,
        'size': size,
        'took': {
          'face': float(face_time),
          'recognition': float(reg_time),
          'all': float(took)
        }
      }
  else:
    return {
        'faces': []
      }

def classify_folder(input_dir, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  nsess = tf.Session()
  image_paths = glob.glob(os.path.join(input_dir, '**/*.png'))
  for index, image_path in enumerate(image_paths):
    result = recognize(image_path, nsess)

    if index % 100 == 0:
      nsess.close()
      tf.reset_default_graph()
      nsess = tf.Session()

    if len(result['faces']) > 0:
      face = result['faces'][0]['face']
      name = ''
      accuracy = 0.0
      for face_name, face_accuracy in face.items():
        name = face_name
        accuracy = face_accuracy
      print("[%i/%i] Result: %s: %.2f, took %.2f" % (index, len(image_paths), name, accuracy, result['took']['all']))

      if face_accuracy > 0.9:
        image = Image.open(image_path)
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        if not os.path.exists(image_output_dir):
          os.makedirs(image_output_dir)
        save_image_path = os.path.join(image_output_dir, os.path.basename(image_path).replace(".png", ".jpg"))
        image.save(save_image_path, format="JPEG")

classify_folder('./learning_incorrect', 'learning/reclassified')
