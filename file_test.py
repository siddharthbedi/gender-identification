import face_recognition
from sklearn import svm
import os
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import uuid
import pickle

filename = "./finalized_model.sav"

# load the model from disk
clf = pickle.load(open(filename, 'rb'))

test_image = face_recognition.load_image_file('C:/Users/HP/Desktop/data/indian-people.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Convert to PIL format
pil_image = Image.fromarray(test_image)
# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
	top, right, bottom, left = face_locations[i]
	test_image_enc = face_recognition.face_encodings(test_image)[i]
	name = clf.predict([test_image_enc])
	naam=str(name)
	print(naam)
	draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
	# Draw label
	text_width, text_height = draw.textsize(naam)
	draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
	draw.text((left + 6, bottom - text_height - 5),naam, fill=(0,0,0))

del draw
# Save image
if no > 0:
	file = "./detected/" + str(uuid.uuid1()) + ".jpg"
	pil_image.save(file)
	print("file saved \n ")

else:
	print("image not saved as no faces are detected, provide proper image \n ")