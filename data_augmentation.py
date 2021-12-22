# Importing necessary libs
import sys
import os
import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import random

#init output
cwd = os.getcwd()
#paths = [cwd+i for i in range(30)]
input_path = cwd +'/input'
output_path = '/Augmented_data'
# Add random noise
def add_noise(img):
    deviation = 50*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# Add bluring noise
def blur(img):
    return (cv2.blur(img,(10,10)))

datagen_bluring = ImageDataGenerator(preprocessing_function= blur)
datagen_zoom = ImageDataGenerator(zoom_range=[0.5,1.0])
datagen_noise = ImageDataGenerator(preprocessing_function=add_noise)
datagen_rot = ImageDataGenerator( rotation_range =60)
datagen_brightness = ImageDataGenerator( brightness_range=[0,2])

datagen = ImageDataGenerator( 
                                 brightness_range=[0.1,1],
                                 zoom_range=[0.5,1.0],
                                 rotation_range = 45,
                                 preprocessing_function=add_noise)

def preprocessing(n_out,gen):
    for root, directories, file in os.walk(input_path):
            for f in file:
                if f.endswith(".png") or f.endswith(".jpg"):
                    
                    img = image.load_img(input_path+'/'+str(f), target_size=(512,512))  #load the image
                    x = image.img_to_array(img) #convert it to a numpy array
                    x = x.reshape((1,) + x.shape) # Reshape it to (1,512,512,3)
                    i = 0
                    for batch in gen.flow(x, batch_size=1, save_to_dir= cwd + output_path, save_prefix='augmented'):
                        i += 1
                        if i % n_out == 0:
                            break

def main():
    print("-----------------------")
    print("1. Add noise")
    print("2. Add bluring")
    print("3. Brightness")
    print("4. Rotation")
    print("5. Zoom")
    print("6. Default")
    print("-----------------------")
    
    arg = input("Select augmentation ")
    n_output = int(input("Number of output images "))
    
    if arg == '1':
          preprocessing(n_output,datagen_noise)
    if arg == '2':
          preprocessing(n_output,datagen_bluring)
    if arg == '3':
          preprocessing(n_output,datagen_brightness)
    if arg == '4':
          preprocessing(n_output,datagen_rot)
    if arg == '5':
          preprocessing(n_output,datagen_zoom)
    if arg == '6':
          preprocessing(n_output,datagen)
    
    print("end!!!!")
if __name__ == "__main__":
    main()

        
    
