import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")

# Read the image
img = cv2.imread("path/to/image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image
cv2.imshow("Faces", img)
cv2.waitKey(0)
