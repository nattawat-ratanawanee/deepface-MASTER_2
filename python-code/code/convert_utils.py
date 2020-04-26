import base64
import numpy as np
import cv2

def numpy_to_Base64_to_string(input):
    return base64.encodestring(base64.b64encode(input)).decode('utf-8')

def string_to_Base64_to_numpy(input):
    return np.frombuffer(base64.decodebytes(base64.decodestring(input.encode("utf-8"))), dtype=np.float32)

def string_to_image(input):
    jpg_original = base64.b64decode(input.encode("utf-8"))
    return cv2.imdecode(np.frombuffer(jpg_original, np.uint8), -1)

def image_to_string(input):
    _, buffer = cv2.imencode('.jpg', input)
    return base64.b64encode(buffer).decode('utf-8')

def byteimage_to_image(input):
    image = input
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def byteimage_to_string(input):
    image = byteimage_to_image(input)
    return image_to_string(image)