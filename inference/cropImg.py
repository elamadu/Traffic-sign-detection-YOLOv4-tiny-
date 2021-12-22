import glob
#from PIL import Image
import os
import cv2

MIN_HEIGHT = 25
MIN_WIDTH = 25
def crop(img,img_height,img_width,detection):
    
    center_x = int(detection[0])
    center_y = int(detection[1])
    box_w = int(detection[2])
    box_h = int(detection[3])
    left = round (center_x - box_w/2)  
    top =  round (center_y - box_h/2)  
    right = round (center_x + box_w/2)  
    bottom = round (center_y + box_h/2)   

    if left < 0:
        left = 0
    if right > img_width:
        right = right
    if top < 0:
        top = 0
    if bottom > img_height:
        bottom = img_height

    img = img[top:bottom,left:right]
    height,width,_ = img.shape
    
    if height >= MIN_HEIGHT and width >= MIN_WIDTH:
        #print('image shape: {}'.format(img.shape))
        img = cv2.resize(img,(50,60), interpolation = cv2.INTER_NEAREST)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else:
        img = None    
    return img
    
