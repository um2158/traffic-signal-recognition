import requests
import cv2
import numpy as np
import cv2
from keras.models import load_model
import sys

url = "http://192.168.0.104:8080/shot.jpg"

FRAME_HEIGHT = 480
FRAME_WIDTH = 540

classes_js = { 0:'Speed limit (20km/h)',
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
            42:'End no passing veh > 3.5 tons' }

model = load_model('traffic_cnn_accuracy_improved.h5',compile=True)

def predict_func(col_image):
    img = cv2.resize(col_image, (50,50))
    imgn = np.array(img).astype('float32')
    imgn /= 255
    imgn = np.asarray(imgn).reshape((1, 50, 50, 3))
    a = model.predict(imgn)
    #results = classes_js[model.predict_classes(imgn)[0].astype(str)]
    results = np.argmax(a,axis=-1)[0]
    return results


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    tex = predict_func(img)

    texte = classes_js[tex]
    img = cv2.putText(img, texte, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Demo",img)
    if cv2.waitKey(1)==27:
        break