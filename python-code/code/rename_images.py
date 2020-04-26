import os
import re
import glob

DATASET_NAME = 'dataset_303'
DATA_DIR = f'C:\\Users\\rattaphon.h\\{DATASET_NAME}\\'
SUFFIX_LIST = ['_passport', '_face_dg2', '_face']
EXTENSIONS = '.jpg'


dir_list = glob.glob(DATA_DIR + '*')
for dir in dir_list:
    id = re.search("dataset_303.(\w+)", dir).group(1)
    for suffix in SUFFIX_LIST:
        file_list = glob.glob(dir + '\\*' + suffix + EXTENSIONS)
        filename = file_list[0]
        new_filename = dir + '\\' + id + suffix + EXTENSIONS
        print(id, 'Origin name:', filename, 'Rename to:', new_filename)
        os.rename(filename, new_filename)

