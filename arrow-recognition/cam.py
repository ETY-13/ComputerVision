# camp.py
# Theng Yang

# contain a basic template for real-time image
# capturing and data feeding to nueral network
# model. 

import time
import cv2
import numpy
from keras.models import model_from_json
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# load trained model
jason_mod = open('net2.json','r')
load_mod = jason_mod.read()
jason_mod.close()
load_net = model_from_json(load_mod)
load_net.load_weights('net2.h5')

print("loaded")

video = cv2.VideoCapture(0)

while True:

    _, frame = video.read()

    cv2.imshow("capture", frame)
    im = Image.fromarray(frame)
    im = im.resize((128,128))
    im.save('catch.jpg')
    im_l = load_img('catch.jpg',target_size=(128,128))
    
    im_array = img_to_array(im_l)
    img_array = numpy.expand_dims(im_array, axis=0)
    im_array = im_array.reshape(1,128,128,3)
 
    pred = int(load_net.predict_classes(im_array))

    print(pred)
    key =cv2.waitKey(1)
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
