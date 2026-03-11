from retinaface import RetinaFace
import cv2
import numpy as np

def extract_face(image):

    faces = RetinaFace.detect_faces(image)

    if len(faces) == 0:
        return None

    key = list(faces.keys())[0]

    x1, y1, x2, y2 = faces[key]['facial_area']

    face = image[y1:y2, x1:x2]

    from PIL import Image
    face_pil = Image.fromarray(face)
    
    return face_pil