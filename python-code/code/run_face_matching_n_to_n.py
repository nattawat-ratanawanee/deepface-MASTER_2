import time
import glob
import flask
import io
import os
import csv
import math

import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
from deepface_engine import DeepFace

deepface_model = 'senet50'
deepface_vgg_model_path = 'models\\vggface\\rcmalli_vggface_tf_notop_senet50.h5'
deepface_mtcnn_model_path = 'models\\mtcnn\\data\\mtcnn_weights.npy'
deepface_pooling = 'avg'
deepface_image_size =  224

_deepface = DeepFace(deepface_model, deepface_pooling, deepface_image_size, deepface_mtcnn_model_path, deepface_vgg_model_path)

INPUT_FOLDER = 'C:\\Users\\rattaphon.h\\dataset_303_extracted\\'
SAVE_FILE = 'result_gen_imp_passport_senet_dataset_303.csv'

FACE_SUFFIX = '_face.jpg'
CARD_SUFFIX = '_face_dg2.jpg'
CARD_SUFFIX_2 = '_passport.jpg'

with open(SAVE_FILE, "w", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(['Selfie', 'Passport', 'Score', 'NormScore'])

list_key = []
list_id = {}
fol_list = glob.glob(INPUT_FOLDER + '*')
for fol in fol_list:
    list_key.append(os.path.basename(fol))
    list_id[os.path.basename(fol)] = []
    list_id[os.path.basename(fol)].append(fol + '\\' + os.path.basename(fol) + FACE_SUFFIX)
    list_id[os.path.basename(fol)].append(fol + '\\' + os.path.basename(fol) + CARD_SUFFIX)
    list_id[os.path.basename(fol)].append(fol + '\\' + os.path.basename(fol) + CARD_SUFFIX_2)

count_match = 0

list_score_gen = []
list_score_imp = []

total = len(list_id) ** 2
progress = 0
time_start = time.time()
for key_selfie, value_selfie in list_id.items():
    for key_idcard, value_idcard in list_id.items():
        FACE_TMP = value_selfie[0]
        CARD_TMP = value_idcard[2]

        list_img = [FACE_TMP, CARD_TMP]
        print(list_img)
        extracted_template = _deepface.get_embeddings(list_img)

        match_result = _deepface.is_match(extracted_template[0], extracted_template[1])

        norm_score = math.acos(match_result[1]) / (np.pi/2)

        with open(SAVE_FILE, "a", newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow([key_selfie, key_idcard, match_result[1], norm_score])

        if key_selfie == key_idcard:
            list_score_gen.append(match_result[1])
        else:
            list_score_imp.append(match_result[1])

        avg_spd_dt = (time.time()-time_start) / (progress + 1) # second per sample
        eta = (total - progress) * avg_spd_dt
        eta_h = math.floor( eta / 3600 )
        eta_m = math.floor((eta % 3600) / 60)
        eta_s = eta % 60

        print(key_selfie, key_idcard, match_result[1], norm_score, "elapsed time = ", (time.time()-time_start),
              'ETA', eta_h, ':', eta_m, ':', eta_s,
              'progress', round((progress/total)*100, 2), '%')
        progress += 1

print("Average elapsed time = ", float((time.time()-time_start)/(len(list_id)**2)))
