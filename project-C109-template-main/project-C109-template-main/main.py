import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("C:/Users/emmaj/OneDrive/Desktop/C110/project-C109-template-main/converted_keras/keras_model.h5")
video=cv2.VideoCapture(0)
while True:
    check,frame=video.read()
    img=cv2.resize(frame,(224,224))
    i1=np.array(img,dtype=np.float32)
    i2=np.expand_dims(i1,axis=0)
    n_img=i2/255.0
    prediction=model.predict(n_img)
    #predict_class=np.argmax(prediction,axis=1)
    #print("prediction:",predict_class)
    cv2.imshow("result",frame)
    key=cv2.waitKey(1)
    if key==32:
     break
video.release()