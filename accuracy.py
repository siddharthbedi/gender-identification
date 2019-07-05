import face_recognition
from sklearn import svm
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('./train/')

print("Running")

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

X_train, X_test, y_train, y_test = train_test_split(encodings, names, test_size=0.2, random_state=42)
print("train_test_split completed \n ")


# Create and train the SVC classifier
print("load dataset completed \n training model started")
clf = svm.SVC(gamma='scale')
clf.fit(X_train,y_train)

print("model training completed \n saving the model")

y_pred = clf.predict(X_test)
#precision = average_precision_score(y_test, y_pred)
#print("average precision score is:")
#print(precision)

train_accuracy = accuracy_score(y_train, clf.predict(X_train))
print("train accuracy:")
print(train_accuracy)

test_accuracy = accuracy_score(y_test, y_pred)
print("test accuracy:")
print(test_accuracy)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))