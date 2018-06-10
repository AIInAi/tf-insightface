'''
Some of below services are referring:
http://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html

'''

import numpy as np
from configs import configs


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
    return face_dist_value


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=configs.face_similarity_threshold):
    true_list = list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    similar_indx = list(np.where(true_list)[0])
    return similar_indx
