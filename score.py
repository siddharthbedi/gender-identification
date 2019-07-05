import face_recognition
from sklearn import svm
import os
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import uuid
import pickle
#from accuracy import *

filename = "./finalized_model.sav"

# load the model from disk
clf = pickle.load(open(filename, 'rb'))

print(type(y_test))
