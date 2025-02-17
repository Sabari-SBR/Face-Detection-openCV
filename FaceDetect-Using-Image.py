import cv2

# Trained DataSet
alg = 'haarcascade_frontalface_default.xml'
dataset = cv2.CascadeClassifier(alg)

# Read the Image
img = cv2.imread('Assets/Images/FD-group.png')

# Convert grayscale
gray_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = dataset.detectMultiScale(gray_Image)

for x, y, w, h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("FaceDetectingImage", img)
cv2.waitKey()