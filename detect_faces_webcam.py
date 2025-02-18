# DETECT FACES WITH WEBCAM

import cv2

def detect_faces_with_webcam():

    # Trained Dataset
    alg = 'models/haarcascade_frontalface_default.xml'
    face_model = cv2.CascadeClassifier(alg)

    # Capture the video
    capture = cv2.VideoCapture(0)  # default webcam id

    if not capture.isOpened():
        return
    
    while True:
        ret, frame = capture.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray_frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Detection with Video", frame)
        key = cv2.waitKey(1)
        if key == 27:  # 27 is a esc key
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_with_webcam()
