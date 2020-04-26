import cv2
import numpy as np
from keras.models import load_model
from face_detection import FaceDetection

class LivenessDetection():
    def __init__(self, face_detection_model_path, liveness_model_path, threshold, image_size):
        self.model = None
        self.liveness_model_path = liveness_model_path
        self.face_detection_model_path = face_detection_model_path
        self.threshold = threshold
        self.image_size = image_size

    def initial(self):
        self.model = load_model(self.liveness_model_path)
        self.model._make_predict_function()
        self.face_detect = FaceDetection(self.face_detection_model_path)

    def process(self, image, mode):
        data = []
        
        x, y, w, h = self.face_detect.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if w > 0 and h > 0:
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = cv2.resize(face_image, (self.image_size, self.image_size))
        else:
            face_image = np.zeros((self.image_size, self.image_size), dtype=int)

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:,:,0] = face_image
        data.append(image)

        data = np.array(data, dtype="float") / 255.0
        preds = self.model.predict(data)
        score = preds[0][0]
        predict = True if score > self.threshold else False
        return score, predict
