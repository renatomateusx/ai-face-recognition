# AI Face Recognition
Project was made as an example for showing in post-graduate presentation

Project made to understand how AI Face recognition works.

# Used:
* OpenCV
* LPBH
* Personal images
* Trained the classifier

# Usage:
* pip install requirements
* choose the file camera_lbph and that is it.

# Record your own pictures
* Take pictures of you.
* Compress it to a YourName.zip
* Run the follow code:

```Python
  import zipfile
  path = '/pathTo/YourName.zip'
  zip_object = zipfile.ZipFile(file=path, mode='r')
  zip_object.extractall('./ToYourPath')
  zip_object.close()
  
  import os
  def get_image_data():
    paths = [os.path.join('/ToYourPathExtract', f) for f in os.listdir('/ToYourPathExtract')]
    faces = []
    ids = []
    for path in paths:
      image = Image.open(path).convert('L')
      image_np = np.array(image, 'uint8')
      id = int(path.split('.')[1])
  
      ids.append(id)
      faces.append(image_np)
  
    return np.array(ids), faces
```
```Python
  ids, faces = get_image_data()
  
  ids, faces
```

 * Creating the Classifier:

```Python
  lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
  lbph_classifier.train(faces, ids)
  lbph_classifier.write('lbph_classifierr.yml')

  lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
  lbph_face_classifier.read('/content/lbph_classifierr.yml')

  paths = [os.path.join('/toYourPath', f) for f in os.listdir('/toYourPath')]
  for path in paths:
    image = Image.open(path).convert('L')
    image_np = np.array(image, 'uint8')
    prediction, _ = lbph_face_classifier.predict(image_np)
    expected_output = int(path.split('.')[1])
  
    cv2.putText(image_np, 'Pred: ' + str(prediction), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
    cv2.putText(image_np, 'Exp: ' + str(expected_output), (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
    cv2_imshow(image_np)
```
 * You're good to go. Get the classifier.yml and past it into your camera_lbph.py folder. Run the code.

![Screenshot](demo.gif)
