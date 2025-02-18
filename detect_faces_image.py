# DETECT FACES IN IMAGE

import cv2

def detect_faces_in_image(image_path):

    # Trained model
    alg = 'models/haarcascade_frontalface_default.xml'
    face_model = cv2.CascadeClassifier(alg)

    # Read the Image
    image = cv2.imread(image_path)

    if image_path is None:
        return

    gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray_Image)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection with Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_in_image("assets/images/face-person.png")
