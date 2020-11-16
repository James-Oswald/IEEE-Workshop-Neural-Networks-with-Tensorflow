
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #disable Tensorflow warning messages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from makeMNISTImage import imageprepare


#====================== Use MNIST Digits ==========================
#(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
#print(train_images.shape)

#plt.matshow(train_images[0, :, :]) 
#plt.show() 

#train_images = np.expand_dims(train_images, -1) #color channels
#train_images = np.expand_dims(train_images, 1) #batch size

#====================== Use Custom Digits ==========================
img = imageprepare("2.png")
plt.matshow(img)
plt.show()

img = np.expand_dims(img, -1) #color channels
img = np.expand_dims(img, 0) #batch size


#==================== Test Model ====================================
model = keras.models.load_model("CnnModel.hdf5")    #Convolutional NN
#model = keras.models.load_model("3b1bModel.hdf5")  #Simple Deep NN

#store the last layers output in result
result = model(img)   
#result = model(train_images[0, :, :, :, :])

for i in range(0, 10):
    print("Is " + str(i) + ": %" + str(100 * result[0][i].numpy()))

