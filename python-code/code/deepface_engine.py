from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2
import numpy as np
import tensorflow as tf

class DeepFace():
    def __init__(self, model_name, pooling, image_size, mtcnn_model_path, vgg_model_path):
        self.model_name = model_name
        self.pooling = pooling
        self.image_size = image_size
        self.mtcnn_model_path = mtcnn_model_path
        self.vgg_model_path = vgg_model_path
        self.model = VGGFace(model=self.model_name, include_top=False, input_shape=(self.image_size, self.image_size, 3), pooling=self.pooling, weights_path=self.vgg_model_path)
        self.detector = MTCNN(weights_file=self.mtcnn_model_path)

    def extract_face(self, img, required_size=(224, 224)):
        results = []
        results = self.detector.detect_faces(img)

        return_data = {}
        return_data["Status"] = False
        return_data["Face"] = np.zeros((224,224,3)).astype(np.uint8)
        return_data["X"] = -1
        return_data["Y"] = -1
        return_data["Width"] = -1
        return_data["Height"] = -1
        if len(results) == 0:
            return return_data

        face_size = 0
        face_id = 0
        for i in range(len(results)):
            x1, y1, width, height = results[i]['box']
            if face_size < width*height:
                face_size = width*height
                face_id = i

        x1, y1, width, height = results[face_id]['box']
        x2, y2 = x1 + width, y1 + height

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > img.shape[1]:
            x2 = img.shape[1]
        if y2 > img.shape[0]:
            y2 = img.shape[0]

        face = img[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        return_data["Status"] = True
        return_data["Face"] = face_array
        return_data["X"] = x1
        return_data["Y"] = y1
        return_data["Width"] = width
        return_data["Height"] = height
        return return_data

    def read_face(self, filename):
        face = pyplot.imread(filename)
        image = Image.fromarray(face)
        face_array = asarray(image)
        return face_array

    def get_embeddings(self, filenames):
        faces = [self.read_face(f) for f in filenames]
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = None
        yhat = self.model.predict(samples)
        return yhat
    
    def is_match(self, known_embedding, candidate_embedding, thresh=0.5):
        score = cosine(known_embedding, candidate_embedding)
        if score <= thresh:
            return (True, score)
        else:
            return (False, score)