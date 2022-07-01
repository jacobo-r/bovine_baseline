
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import math
from sklearn import linear_model



class VideoClassifier(object):


    def __init__(self):

        self.IMG_SHAPE = (
            224,
            224,
            3,
        )  

        ############# Feature extractor #############
        feature_extractor = tf.keras.applications.MobileNetV2(
                input_shape=self.IMG_SHAPE, include_top=False, weights="imagenet"
            )
        feature_extractor.trainable = False
        inputs = tf.keras.Input(shape=self.IMG_SHAPE)
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
        x = preprocess_input(inputs)
        x = feature_extractor(x, training=False)
    
        x = global_average_layer(x)
        outputs = tf.keras.layers.Dropout(0.2)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["sparse_categorical_accuracy"],
        )

        self.feature_extractor= model

        ############# Classification model #############

        self._model=linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')


    def resize_frame(self, frame):        
        resized_img=Image.fromarray(frame).resize((224,224))
        return np.array(resized_img)

        # function used to generate the image datsets limited by the pred_time
    def gen_videos(self,videolist):

        newvideos=[] 
        for video in videolist:
            reducedvideo= video.read_frame(video.frame_times[-1]) #takes the last possible frame 
            #resize from 250 to 224
            reducedvideo=self.resize_frame(reducedvideo)    
            newvideos.append(reducedvideo)
        return newvideos

    def class_to_int(self,argument):
        switcher = {
        'A':0,
        'B':1,
        'C':2,
        'D':3,
        'E':4,
        'F':5,
        'G':6,
        'H':7,
        }
        return switcher.get(argument, "nothing")

    def int_to_class(self,argument):
        switcher = {
        0:'A',
        1:'B',
        2:'C',
        3:'D',
        4:'E',
        5:'F',
        6:'G',
        7:'H',
        }
        return switcher.get(argument, "nothing")


    #################################################################################################
    def fit(self, videos: list, y, pred_time: float):
        X_for_classifier= np.array(videos)
        y_for_classifier= y
        X_for_classifier= np.array(self.gen_videos(X_for_classifier))
        X_for_classifier = np.repeat(X_for_classifier[...,np.newaxis], 3, -1)

        func=np.vectorize(self.class_to_int) #from labels to int
        train_labels=func(y_for_classifier) 

        #extracting the features 
        featmap=[]
        for i in range(len(videos)):  # num of videos in ds
            f1=self.feature_extractor.predict(X_for_classifier[i][None,:,:], verbose='false') # ith video
            featmap.append(f1)

        #training the model
        X_train= np.array(featmap).squeeze()
        y_train=np.array(train_labels)
        self._model.fit(X_train, y_train)

        return self

    #################################################################################################


    def predict(self, videos: list, pred_time: float):

        Xtest_for_classifier= np.array(videos)
        Xtest_for_classifier= np.array(self.gen_videos(Xtest_for_classifier))
        Xtest_for_classifier = np.repeat(Xtest_for_classifier[...,np.newaxis], 3, -1)
      
        test_featmap=[]
        for i in range(len(videos)):
            f1=self.feature_extractor.predict(Xtest_for_classifier[i][None,:,:],  verbose='false')
            test_featmap.append(f1)
 
        fmap= np.array(test_featmap).squeeze()
        #generating the probability predictions used by the WeightedClassificationError
        predproba= self._model.predict_proba(fmap)

        return predproba