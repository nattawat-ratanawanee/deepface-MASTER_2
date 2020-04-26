import pandas as pd
from matplotlib import pyplot
import numpy as np
import time
import math

data = pd.read_csv('result_gen_imp_passport_senet_dataset_303.csv')
FAR1_val = 0.01
FAR2_val = 0.001

# mode 0 = less score is better
# mode 1 = higher score is better
mode = 1

list_score_gen = []
list_score_imp = []
# actual_threshold = 0.77154

for i in range(len(data)):
    score = data.iloc[i,2]
    # score = math.acos(data.iloc[i,2]) / (np.pi/2)
    # tmp_score = math.acos(data.iloc[i,2]) / (np.pi/2)
    # score = tmp_score / (actual_threshold*2) if tmp_score <= actual_threshold else ((tmp_score - actual_threshold) * 0.5 / (1-actual_threshold)) + 0.5
    if data.iloc[i,0] == data.iloc[i,1]:
        list_score_gen.append(score)
    else:
        list_score_imp.append(score)

list_score_gen = np.array(list_score_gen)
list_score_imp = np.array(list_score_imp)

time_st = time.time()

list_FAR = []
list_FRR = []
list_thresh = np.linspace(0, 1, 999)
# list_thresh = np.linspace(-1, 100, 102)
# list_thresh = [0.292]

for THRESH in list_thresh:
    if mode == 0:
        count_TP = np.sum((list_score_gen <  THRESH ).astype(int))
        count_TN = np.sum((list_score_imp >= THRESH ).astype(int))
        count_FP = np.sum((list_score_imp <  THRESH ).astype(int))
        count_FN = np.sum((list_score_gen >= THRESH ).astype(int))
    else:
        count_TP = np.sum((list_score_gen >  THRESH ).astype(int))
        count_TN = np.sum((list_score_imp <= THRESH ).astype(int))
        count_FP = np.sum((list_score_imp >  THRESH ).astype(int))
        count_FN = np.sum((list_score_gen <= THRESH ).astype(int))

    print("TP", count_TP, "FP", count_FP, "FN", count_FN, "TN", count_TN, )
    list_FAR.append(count_FP / (count_FP+count_TN))
    list_FRR.append(count_FN / (count_FN+count_TP))

list_FAR = np.array(list_FAR)
list_FRR = np.array(list_FRR)

print('Time: ', time.time()-time_st)

if mode == 0:
    FAR1_idx = np.sum((list_FAR < FAR1_val ).astype(int))
    FAR2_idx = np.sum((list_FAR < FAR2_val ).astype(int))
else:
    FAR1_idx = np.sum((list_FAR > FAR1_val ).astype(int))
    FAR2_idx = np.sum((list_FAR > FAR2_val ).astype(int))

buf1 = "FRR = %.5f\nFAR = %.5f\nat TH = %.5f" % (list_FRR[FAR1_idx], FAR1_val, list_thresh[FAR1_idx])
print(buf1)
buf2 = "FRR = %.5f\nFAR = %.5f\nat TH = %.5f" % (list_FRR[FAR2_idx], FAR2_val, list_thresh[FAR2_idx])
print(buf2)

pyplot.plot(list_thresh, list_FAR, color='g', label='FAR')
pyplot.plot(list_thresh, list_FRR, color='r', label='FRR')

pyplot.axvline(x=list_thresh[FAR1_idx], color='b', ls='--', label=buf1)
pyplot.axvline(x=list_thresh[FAR2_idx], color='k', ls='--', label=buf2)

pyplot.legend(loc="upper right")

pyplot.title("EER Selfie vs Passport - Deepface")
pyplot.xlabel("Threshold")
pyplot.ylabel("Error Rate")
pyplot.show()
