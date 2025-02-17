import cv2

# Trained Dataset
alg = 'haarcascade_frontalface_default.xml'
dataset = cv2.CascadeClassifier(alg)

# Redd the Video file
video = cv2.VideoCapture('Assets/Videos/FD-Video.mp4')

while True:
    success, frame = video.read()

    if success == True:
        gray_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray_Image)

        # Draw a rectangle to the faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("FaceDetectingVideo", frame)
        key = cv2.waitKey(1)
        if key == 27:   # esc key
            break
    else:
        break