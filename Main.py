from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import copy
import itertools
import os
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import time

global filename
global model, sc
global accuracy, precision, recall, fscore
min_detection_confidence = 0.7
min_tracking_confidence = 0.5
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
drawingModule = mp.solutions.drawing_utils

min_detection_confidence = 0.7
min_tracking_confidence = 0.5

global X, Y, labels
global X_train, X_test, y_train, y_test

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def uploadDataset():
    global filename, labels, X, Y, dataset
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    counter = 0
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
                counter += 1
    
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):        
                name = os.path.basename(root)
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.flip(img, 1)
                debug_image = copy.deepcopy(img)
                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        pre_processed_landmark_list = np.asarray(pre_processed_landmark_list)
                        X.append(pre_processed_landmark_list)
                        label = getLabel(name)
                        Y.append(label)  
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    print(np.unique(Y, return_counts=True))    
    text.insert(END,"Hand Gesture Class labels found in Dataset : "+str(labels)+"\n\n")    
    text.insert(END,"Total Hand Gesture found in dataset : "+str(counter))

def processDataset():
    global X, Y, sc
    text.delete('1.0', END)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Hand Gesture Features Shuffling & Normalization processing Completed")    

def splitDataset():
    global X_train, X_test, y_train, y_test
    global X, Y
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% Hand Gesture Features used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% Hand Gesture Features used for testing : "+str(X_test.shape[0])+"\n\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")      

def trainRandomForest():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    calculateMetrics("Random Forest", y_test, predict)

def trainCNN():
    global X_train, X_test, y_train, y_test, model
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    model = Sequential()
    model = Sequential()
    model.add(Convolution2D(32, (1, 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    model.add(Flatten())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/weights.hdf5', verbose = 1, save_best_only = True)
        hist = model.fit(X_train1, y_train1, batch_size = 8, epochs = 150, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        model.load_weights("model/weights.hdf5")
    predict = model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    calculateMetrics("CNN", y_test, predict)

def gesturefromVideo():
    global model, sc, min_detection_confidence, min_tracking_confidence, labels
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "testVideos")
    camera = cv2.VideoCapture(filename)
    text.insert(END,"Hand Gesture to Words\n\n")
    while(True):
        (grabbed, frame) = camera.read()
        if frame is not None:
            frame = cv2.resize(frame, (1000, 800))
            img = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(img)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    data = []
                    data.append(pre_processed_landmark_list)
                    data = np.asarray(data)
                    data = sc.transform(data)
                    data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
                    predict = model.predict(data)
                    predict = np.argmax(predict)
                    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
                    text.insert(END,labels[predict]+"\n")
                    text.update_idletasks()
            img = cv2.resize(img, (700, 600))        
            cv2.imshow("Sign Langauge Prediction", img)
            time.sleep(1)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

def gesturefromWebcam():
    text.delete('1.0', END)
    text.insert(END,"Hand Gesture to Words\n\n")
    global model, sc, min_detection_confidence, min_tracking_confidence, labels
    camera = cv2.VideoCapture(0)
    while(True):
        (grabbed, frame) = camera.read()
        if frame is not None:
            frame = cv2.resize(frame, (1000, 800))
            img = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(img)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    data = []
                    data.append(pre_processed_landmark_list)
                    data = np.asarray(data)
                    data = sc.transform(data)
                    data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
                    predict = model.predict(data)
                    predict = np.argmax(predict)
                    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
                    text.insert(END,labels[predict]+"\n")
                    text.update_idletasks()
            cv2.imshow("Sign Langauge Prediction", img)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()     

def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','F1 Score',fscore[0]],['Random Forest','Accuracy',accuracy[0]],
                       ['CNN','Precision',precision[1]],['CNN','Recall',recall[1]],['CNN','F1 Score',fscore[1]],['CNN','Accuracy',accuracy[1]],                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar')
    plt.show()

main = tkinter.Tk()
main.title("Transforming Hand Gesture into Words. A Novel Approach for Assistive Communications using ML Techniques")
main.geometry("1300x1200")

font = ('times', 16, 'bold')
title = Label(main, text='Transforming Hand Gesture into Words. A Novel Approach for Assistive Communications using ML Techniques')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Hand Gesture Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

splitButton = Button(main, text="Train & Test Split", command=splitDataset)
splitButton.place(x=700,y=250)
splitButton.config(font=font1) 

svmButton = Button(main, text="Train Machine Learning Random Forest Algorithm", command=trainRandomForest)
svmButton.place(x=700,y=300)
svmButton.config(font=font1)

cnnButton = Button(main, text="Train Deep Learning CNN Algorithm", command=trainCNN)
cnnButton.place(x=700,y=350)
cnnButton.config(font=font1)

videoButton = Button(main, text="Hand Gesture to words from Video", command=gesturefromVideo)
videoButton.place(x=700,y=400)
videoButton.config(font=font1)

webcamButton = Button(main, text="Hand Gesture to words from Webcam", command=gesturefromWebcam)
webcamButton.place(x=700,y=450)
webcamButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=700,y=500)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
