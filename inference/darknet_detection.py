import sys
sys.path.append('/home/amadou/Downloads/inferenz-test/darknet/')
import darknet
from darknet import load_network
import cv2
import time
import classifier
#import classifier_rt
from classifier import Classifier
#from classifier_rt import ClassifierRT
from cropImg import crop


cnn = Classifier('classifier_last.h5')
#cnn = ClassifierRT('tensorrt_model_last')
weights = 'yolov4-tiny_training_best.weights'
datafile = 'darknet/data/obj.data'
cfg = 'yolov4-tiny_testing.cfg'
thresh= 0.65
network, class_names, class_colors = load_network(cfg, datafile,  weights, 1)
width = darknet.network_width(network)
height = darknet.network_height(network)

    


def image_detection(image):
    #image = cv2.imread(image_path)
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())   
    #while(True):
    #print('---------------- Next loop --------------------')
    start = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print('==========================================')
    #print("YOLO FPS: {}".format(1.0/(time.time()-start)))
    #print(detections)   
    #image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) 
    output = []
    #print(detections)
    for sign in detections:
        _,conf,box = sign
        croped_img = crop(image_resized,height,width,box)
        #cv2.imshow('Image',image_resized)
        #key = cv2.waitKey(1000)
        #newstart = time.time() 
        if croped_img is not None:
            category,certainty = cnn.predict(croped_img)
            #print("CNN FPS: {}".format(1.0/(time.time()-newstart)))
            #output.append((classes[category],certainty,box))
            output.append((category,certainty,box))
    #print(output)   
    #print("Total FPS: {}".format(1.0/(time.time()-start)))
    #print('==========================================')

    darknet.free_image(darknet_image)
    #image = darknet.draw_boxes(detections, image_resized, class_colors)
    return output


