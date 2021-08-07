import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob


from tensorflow import keras
model = keras.models.load_model('my_model1.h5')

default_image_size = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print("Error : "+e)
        return None


cv_img = "test.JPG"



n= cv2.imread(cv_img)
n_img = cv2.resize(n, default_image_size)
np_images = np.array(n_img)
np_image = np.array(np_images, dtype=np.float16) / 225.0

np_image = np.array([np_image])

data =  model.predict(np_image)

print(data)
data_show = np.round(data, 2)
print(data_show)
i = np.argmax(data_show, axis=1)
if i == 0:
    print("Pepper bell Bacterial spot")
if i == 1:
    print("Pepper bell healthy")
if i == 2:
    print("Potato Early blight")
if i == 4:
    print("Potato healthy")
if i == 3:
    print("Potato Late blight") 