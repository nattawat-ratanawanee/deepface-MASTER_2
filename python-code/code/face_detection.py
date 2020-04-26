import numpy as np
import cv2

class FaceDetection():
    global face_cascade

    def __init__(self, model_path):
        self.face_cascade = cv2.CascadeClassifier(model_path)
        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            max_w = 0
            max_index = -1
            for index, face in enumerate(faces):
                x, y, w, h = face
                if w > max_w:
                    max_w = w
                    max_index = index

            x, y, w, h = faces[max_index]
            return x, y, w, h
        return 0, 0, 0, 0