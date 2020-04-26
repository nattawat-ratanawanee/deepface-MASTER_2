import cv2
import numpy as np
from tensorflow import keras
from face_detection import FaceDetection
import imutils

class LivenessDetection():
    def __init__(self, face_detection_model_path, liveness_model_path, threshold, image_size):
        self.model = None
        self.liveness_model_path = liveness_model_path
        self.face_detection_model_path = face_detection_model_path
        self.threshold = threshold
        self.image_size = image_size

    def initial(self):
        self.model = keras.models.load_model(self.liveness_model_path)
        self.model._make_predict_function()
        self.face_detect = FaceDetection(self.face_detection_model_path)

    def process(self, image, mode):
        data = []
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = imutils.resize(image, width=800)
        x, y, w, h = self.face_detect.process(image)

        if w > 0 and h > 0:
            # img_face = image[y:y+h, x:x+w]
            new_pad = np.min([x, image.shape[1]-(x+w), y, image.shape[0]-(y+h)])
            image = image[y-new_pad:y+h+new_pad, x-new_pad:x+w+new_pad,:]
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) /127.5 - 1

            score = self.model.predict_on_batch(np.expand_dims(image, axis=0))[0]
            predict = True if score > self.threshold else False
        else:
            score = -999
            predict = False
        return score, predict
