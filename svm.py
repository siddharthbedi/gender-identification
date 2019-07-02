import face_recognition
from sklearn import svm
import os

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('C:/Users/HP/face_recognition_examples/train/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("C:/Users/HP/face_recognition_examples/train/" + person)
    #print(pix)
    
    # Loop through each training image for the current person
    for person_img in pix:
        try:
            face = face_recognition.load_image_file("C:/Users/HP/face_recognition_examples/train/" + person + "/" + person_img)
            face_enc = face_recognition.face_encodings(face)[0]
            
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        
        except:
            pass
# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

'''# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('C:/Users/HP/face_recognition_examples/img/Dhwani/Photos/6.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found: \n")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)'''