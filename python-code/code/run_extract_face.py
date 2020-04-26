import time
import glob
import os

import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
# from convert_utils import *
from deepface_engine import DeepFace

deepface_model = 'senet50'
deepface_vgg_model_path = 'models\\vggface\\rcmalli_vggface_tf_notop_senet50.h5'
deepface_mtcnn_model_path = 'models\\mtcnn\\data\\mtcnn_weights.npy'
deepface_pooling = 'avg'
deepface_image_size =  224

FACE_SUFFIX = '_face.jpg'
CARD_SUFFIX = '_face_dg2.jpg'
CARD_SUFFIX_2 = '_passport.jpg'

_deepface = DeepFace(deepface_model, deepface_pooling, deepface_image_size, deepface_mtcnn_model_path, deepface_vgg_model_path)

INPUT_FOLDER = 'C:\\Users\\rattaphon.h\\dataset_303\\'
SAVE_FOLDER = 'C:\\Users\\rattaphon.h\\dataset_303_extracted\\'

list_key = []
list_id = {}
fol_list = glob.glob(INPUT_FOLDER + '*')
for fol in fol_list:
    if not os.path.exists(fol.replace(INPUT_FOLDER, SAVE_FOLDER)):
        os.makedirs(fol.replace(INPUT_FOLDER, SAVE_FOLDER))

    print("ID", os.path.basename(fol))
    list_key.append(os.path.basename(fol))
    list_id[os.path.basename(fol)] = []
    list_id[os.path.basename(fol)].append(fol + '\\' + os.path.basename(fol) + FACE_SUFFIX)
    list_id[os.path.basename(fol)].append(fol + '\\' + os.path.basename(fol) + CARD_SUFFIX)
    list_id[os.path.basename(fol)].append(fol + '\\' + os.path.basename(fol) + CARD_SUFFIX_2)

for key, value in list_id.items():
    print(key)
    for i in range(3):
        print("File path", value[i])
        img = cv2.cvtColor(cv2.imread(value[i]), cv2.COLOR_BGR2RGB)
        face_selfie = _deepface.extract_face(img)
        Image.fromarray(face_selfie['Face']).save(value[i].replace(INPUT_FOLDER, SAVE_FOLDER))
