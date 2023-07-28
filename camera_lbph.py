import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbph_classifierr.yml")
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    connected, image = camera.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(30,30))
    for (x, y, w, h) in detections:
        image_face = cv2.resize(image_gray[y:y + w, x:x + h], (width, height))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
        id, confidence = face_recognizer.predict(image_face)
        name = ""
        # ID IS 9 BECAUSE THE IMAGE HAS SOMETHING LIKE THIS: RENATO.9.jpg. The right thing to do is the server
        # process the image and save it as the same as the ID key for searching the user.
        # When you get the ID you can fetch on database and get the right information from the user.
        # This is used for fetching criminal records information and identify the criminal face.
        if id == 9:
            name = 'Renato'

        cv2.putText(image, name, (x,y +(w+30)), font, 2, (0,255,255))
        cv2.putText(image, str(confidence), (x,y + (h+50)), font, 1, (0,255,255))

    cv2.imshow("Face", image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
