# camp.py
# Theng Yang

# contain a basic template for real-time image
# capturing and data feeding to nueral network
# model. 

import pygame as pg

import time
import cv2
import numpy
from keras.models import model_from_json
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# load trained model
bin_mod = open('binet.json','r')
class_mod = open('net2.json','r')
load_bin = bin_mod.read()
load_class =class_mod.read()
bin_mod.close()
class_mod.close()

load_bin = model_from_json(load_bin)
load_bin.load_weights('binet.h5')

load_class = model_from_json(load_class)
load_class.load_weights('net2.h5')

print("loaded")

video = cv2.VideoCapture(0)

# initialize pygame

pg.init()
window = pg.display.set_mode((400,400))
window.fill((255,255,255))
pg.display.set_caption("drive sim")
loop =True

pos_x = 200
pos_y = 200
box_height = 50
box_width = 30
box_color = (90,90,90)

while loop:
    
    pg.time.delay(50)

    _, frame = video.read()
    cv2.imshow("capture", frame)
    im = Image.fromarray(frame)
    im = im.resize((128,128))
    im.save('catch.jpg')
    im_l = load_img('catch.jpg',target_size=(128,128))
    
    im_array = img_to_array(im_l)
    img_array = numpy.expand_dims(im_array, axis=0)
    im_array = im_array.reshape(1,128,128,3)
 

    for event in pg.event.get():
        if event.type == pg.QUIT:
            loop = False

    bin_pred = int(load_bin.predict_classes(im_array))

    if(bin_pred ==1):
        class_pred = int(load_class.predict_classes(im_array))
       
        if(class_pred == 0):
            if(pos_y <400-box_height):
                pos_y +=5
        elif(class_pred==1): 
             if(pos_x >0+box_width):
                pos_x -=5
        elif(class_pred == 2):  
            if(pos_x <400-box_height):
                pos_x +=5
        elif(class_pred ==3):
             if(pos_y >0+box_width):
                 pos_y -=5
    else:
        print("No arrow")
 
    window.fill((255,255,255))
    pg.draw.rect(window,box_color, (pos_x,pos_y, box_height, box_width))
    pg.display.update()

    key =cv2.waitKey(1)
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()

