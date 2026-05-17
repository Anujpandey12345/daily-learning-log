import cv2
import pickle, os
from utils import get_face_landmarks



save_model = os.path.join("Day6_emotation_detection", "model")
emotion = ['HAPPY', 'SAD', 'SURPRISED']
with open(save_model, "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)


rat, frame = cap.read()

while rat:
    rat, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, static_image_mode=False)
    output = model.predict([face_landmarks])
    cv2.putText(frame, emotion[int(output[0])], (10 ,frame.shape[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    cv2.imshow('frame', frame)
    cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()
