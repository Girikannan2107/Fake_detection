import cv2
from mtcnn import MTCNN

detector = MTCNN()

def extract_face(image):

    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']
    face = image[y:y+h, x:x+w]

    face = cv2.resize(face, (224,224))

    return face