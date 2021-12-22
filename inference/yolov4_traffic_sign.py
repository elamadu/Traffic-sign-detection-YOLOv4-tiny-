import cv2
import numpy as np
import time
import darknet_detection
import cropImg


img = cv2.imread('test_105.png')
img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
detections = darknet_detection.image_detection(img)
print(detections)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
colors = np.random.uniform(0,255, size = (len(detections),3))
i = 0
if len(detections) > 0:

    for detection in detections: 
        c_x,c_y,w,h = detection[2]
        
        w = int(w)
        h = int(h)
        x = int(c_x - w/2)
        y = int(c_y - h/2)
        
        label = str(detection[0])
        confidence = str(detection[1])
        color = colors[i]
        i+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label+" "+confidence,(x,y+0),font,1,color,2)
    cv2.imshow('Image',img)
    #time.sleep(2.4)
    key = cv2.waitKey(10000)
    #if key== 'q':
    #    break
