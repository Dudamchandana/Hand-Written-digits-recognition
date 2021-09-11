import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
'''def load_model1():   
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.built = True
    model.load_weights('./digit_trained.hdf5')
    return model'''
model = load_model('./digit_trained.h5')
run = False
ix,iy = -1,-1
follow = 25
img = np.zeros((512,512,1))
### func
def draw(event, x, y, flag, params):
    global run,ix,iy,img,follow
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if run == True:
            cv2.circle(img, (x,y), 20, (255,255,255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x,y), 20, (255,255,255), -1)
        gray = cv2.resize(img, (28, 28))
        gray = gray.reshape(1, 784)
        result = np.argmax(model.predict(gray))
        result = 'cnn : {}'.format(result)
        cv2.putText(img, org=(25,follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
        follow += 25
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512,512,1))
        follow = 25
### param
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)
while True:    
    cv2.imshow("image", img) 
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
