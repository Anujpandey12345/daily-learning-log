import os, pickle
from PIL import Image
import cv2
from utils import get_face_landmarks
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "..", "Dataset")
save_path = os.path.join("Day6_emotation_detection", "data.txt")

output = []
for emotion_idx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)
        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_idx))
            output.append(face_landmarks)

np.savetxt(save_path, np.asarray(output))


        