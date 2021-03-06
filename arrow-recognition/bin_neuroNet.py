
# Theng Yang
# bin_neuroNet.py

# use neural network template
# to train a binary neural network.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import np_utils
import numpy
import matplotlib.pyplot as plt

def net_model():
    model = Sequential()
    model.add(Convolution2D(30,5,5,input_shape=(128,128,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(15,3,3,activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
    return model

def save(net,name):
    save_model = net.to_json()
    with open(str(name)+".json", "w") as json_file:
        json_file.write(save_model)

    net.save_weights(str(name)+".h5")

    print("saved")


seed = 7
numpy.random.seed(seed)
training = ImageDataGenerator(rescale=1.0/255, width_shift_range=0.0,height_shift_range=0.0,shear_range=0.2,zoom_range=0.2,zca_whitening=True)
train_set = training.flow_from_directory("data/train",target_size=((128,128)),batch_size=278,class_mode="binary")
validating = ImageDataGenerator(rescale=1.0/255)
val_set = validating.flow_from_directory("data/validation",target_size=((128,128)),batch_size=278,class_mode="binary")

x_train,y_train = train_set.next()
x_val, y_val = val_set.next()

#plt.imshow(x_train[63])  # plot the image
#plt.show()              # show the image

x_train = x_train.reshape(x_train.shape[0],128,128,3)
x_val = x_val.reshape(x_val.shape[0],128,128,3)

y_train2 = np_utils.to_categorical(y_train)
y_val2 = np_utils.to_categorical(y_val)


net = net_model()
net.fit(x_train, y_train, validation_data = (x_val, y_val),nb_epoch=10,batch_size=10)
score = net.evaluate(x_val,y_val,verbose = 0)

print("Evaluation score:",score[1] *100.0)
