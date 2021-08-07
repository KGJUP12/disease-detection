from flask import Flask,render_template,request
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


import cv2
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import set_session


import base64




class SomeObj():
    def __init__(self):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.model = keras.models.load_model('my_model.h5')
    def sendM(self):
        return self.sess,self.model,self.graph

global_obj = SomeObj()

default_image_size = tuple((320, 320))




app = Flask(__name__)


@app.route('/index')    #http://127.0.0.1:5000/index
def index():
   return render_template('getimage.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
    ##########################################################################
    # get image uploaded by user
    photo = request.files['file']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    
    ##########################################################################
    #convert image to base64 to user back
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    data = str(jpg_as_text)
    data_image = str(jpg_as_text)

    ##########################################################################

    # resize image into 256x256 and convert image to numpy

    n_img = cv2.resize(img, (256, 256))
    np_images = np.array(n_img)
    np_image = np.array(np_images, dtype=np.float16) / 225.0

    np_image = np.array([np_image])

    ##########################################################################

    # using image numpy array evaluate image by using trained model
    # model will get number data as output

    sess,model,graph = global_obj.sendM()
    with graph.as_default():
        set_session(sess)
        data =  model.predict(np_image)

    print(data.argmax())
    data_show = np.round(data, 2)
    print(data_show)

    ##########################################################################

    # model data ( number ) will be converted to string label

    if data.argmax() == 0:
        class_data = 'Pepper bell Bacterial spot'
    elif data.argmax() == 1:
        class_data = 'Pepper bell healthy'
    elif data.argmax() == 2:
        class_data = 'Potato Early blight'
    elif data.argmax() == 4:
        class_data = 'Potato healthy'
    else:
        class_data = 'Potato Late blight'

    
    # return render_template('result.html',data=data_image[2:-1], class_data=str(list_data[data.argmax()]))
    ##########################################################################

    # send html page with base64 image and string label

    return render_template('result.html',data=data_image[2:-1], class_data=class_data)


if __name__ == '__main__':
   app.run(debug = True)

