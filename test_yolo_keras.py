from ultralytics import YOLO # Ultralytics company package
from tensorflow.keras.models import load_model #trained model weights can be loaded using this .format can be anything h5, hdf5
import cv2 #alias opencv-python used for image related operations
import numpy as np #numerical operations : array conversion etc
import time
import requests #used to get image frame by frame from IP webcam server as base64 string 

sign_classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons'}

warning_classes = {
      0:"Prohibitory",
      1:"Danger",
      2:"Mandatory",
      3:"Other"}

model = YOLO("traffic_yolo.pt")  # YOLO :: load a pretrained model (recommended for training) #for traffic signal detection from entire frame
check_mod = load_model("traffic_classifier.h5") # CUSTOM CNN :: for sign detection in YOLO detected ROI

def predictor(i):
    results = model(i,verbose=False) #verbose True shows process loading bar, time taken for process
    #results is a list of predictions (input frame can have multiple signals)
    for result in results:
        imgg = result.orig_img
        boxes = result.boxes #bounding boxes of predictions
        #print(result)
        for box in boxes:  # returns one box
            coords = list(map(int,box.xyxy[0])) #we convert every value as integer because cv2 only supports integer cropping
            x1,y1,x2,y2 = coords
            img = imgg[y1:y2,x1:x2] #crop image around bounding box area
            img = cv2.resize(img, (30, 30)) # resize into 30*30 size which is input size of Custom CNN model

            pred = np.argmax(check_mod.predict(np.array([img]),verbose=0), axis=-1) #List of 43 values, each index corresponds to a class 
            #for example result returns a list of length 43 with values [0 0 0.5 0.3 0.1 ..... 0.99]
            #max value is 0.99 which means 99% confidence, and this value is present at 43rd index of the list, which corresponds to 'End no passing veh > 3.5 tons' in the original encoded series
            res = sign_classes[pred[0]]
            #corresponding sign in sign_classes dictionary -> sign_classes[43] = 'End no passing veh > 3.5 tons'
            resres = warning_classes[int(box.cls[0])]
            #same for warning_classes dictionary

            #to display a color bounding box around image
            if resres=="Prohibitory":
                i = cv2.rectangle(i, (x1,y1), (x2,y2), prohibitory_color, thickness)
            elif resres=="Danger":
                i = cv2.rectangle(i, (x1,y1), (x2,y2), danger_color, thickness)
            elif resres=="Mandatory":
                i = cv2.rectangle(i, (x1,y1), (x2,y2), mandatory_color, thickness)
            else:
                i = cv2.rectangle(i, (x1,y1), (x2,y2), blue_color, thickness)

            #put text (sign name) on top left of bounding box
            cv2.putText(i, res, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return i

#IP WEbcam url
url = "http://192.168.1.16:8080/shot.jpg"

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
blue_color = (255, 0, 0)
danger_color = (0, 0, 255)
prohibitory_color = (0, 255, 255)
mandatory_color = (0, 255, 0)
thickness = 3

while True: # while the IP webcam server is running run the following code, else break
    try:
        #IT WILL RUN THIS CODE
        img_resp = requests.get(url)
        #this gets a frame from the IP WEBCAM server as a base64 string, which is then converted to a bytes format array and then to a 1D numpy image array
        img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
        img = cv2.imdecode(img_arr,-1)
        res = predictor(img)
        cv2.imshow("Traffic Signal Guidance System",img)
        if cv2.waitKey(1)==27:#press Q to quit gracefully
            break
    except Exception as e:
        #IT WILL RUN THIS EXCEPTION AND PREVENT CRASHING
        print("CONNECTION FAILED",e)