import numpy as np
import cv2
import os
import glob

# list_file = os.listdir('D:/sampling image/Data Set Liveness/Video/Test/attack/Fixed/adverse')
# for filepath in list_file:
#     print(filepath)

def extract_multiple_videos(intput_filenames):
    """Extract video files into sequence of images.
       Intput_filenames is a list for video file names"""

    try:

        # creating a folder named data
        if not os.path.exists('data_test_fixed_adverse'):
            directory = 'data_test_fixed_adverse'
            parent_dir = "D:/sampling image/Data Set Liveness/Video/Test/attack/Fixed/adverse"
            path = os.path.join(parent_dir, directory)
            os.makedirs(path)

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    i = 1  # Counter of first video

    # Iterate file names:
    for intput_filename in intput_filenames:
        cap = cv2.VideoCapture(intput_filename)

        # Keep iterating break
        while True:
            ret, frame = cap.read()  # Read frame from first video

            if ret:

                name = './data_test_fixed_adverse/attack_fix_adverse_' + str(i)  + '.jpg'
                #print('Creating...' + name)

                cv2.imwrite(name, frame)  # Write frame to JPEG file (1.jpg, 2.jpg, ...)
               # cv2.imshow('frame', frame)  # Display frame for testing
                i += 1 # Advance file counter
                if i % 10 == 0:
                    break
            else:
                # Break the interal loop when res status is False.
                break

            cv2.waitKey(100) #Wait 100msec (for debugging)

        cap.release() #Release must be inside the outer loop
        cv2.destroyAllWindows()


list_file = os.listdir('D:/sampling image/Data Set Liveness/Video/Test/attack/Fixed/adverse')
input_filename = {}
for o in list_file:
    # print(o)
    input_filename[o] = o

# filename1 = 'Attack Highdef Client009 Session01 Highdef Photo Adverse-1.mp4'
# filename2 = 'Attack Highdef Client011 Session01 Highdef Photo Adverse-5.mp4'
#
# input_filename = [filename1,filename2]
#
extract_multiple_videos(input_filename)



