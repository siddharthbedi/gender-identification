import face_recognition
from sklearn import svm
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
y
# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('./train/')

print("running")

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("./train/" + person)
    #print(pix)
    
    # Loop through each training image for the current person
    for person_img in pix:
        try:
            face = face_recognition.load_image_file("./train/" + person + "/" + person_img)
            face_enc = face_recognition.face_encodings(face)[0]
            
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        
        except:
            pass
# Create and train the SVC classifier
print("load dataset completed \n training model started")
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

print("model training completed \n saving the model")
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))