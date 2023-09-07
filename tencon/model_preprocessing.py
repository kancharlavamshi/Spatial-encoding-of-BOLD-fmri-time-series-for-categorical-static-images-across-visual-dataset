import pandas as pd
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold


from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
from scipy import interp
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pyts.image import GramianAngularField




def res_ind(df):
  df=df.reset_index()
  df=df.drop(['index'], axis=1)
  return df

def Model_1():
  model = Sequential()
  model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(16, 16,1)) )
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (2, 2), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dense(8,activation='relu'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',
                metrics=['accuracy','Precision','Recall'])
  return model


def Kfloding(kf,model,x,y,floder_path,model_file,subject,epochs,batch_size):
  callback_2 = ModelCheckpoint(floder_path +str(subject)+'__'+str(model_file)+'.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

  cv = StratifiedKFold(n_splits=int(kf),shuffle=True,random_state=98)
  accuracy_v=[]
  y_true=[]
  y_predict = []
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0,1,100)
  i = 1
  for train_index,test_index in cv.split(x,y):
      x_train,x_test=x[train_index],x[test_index]
      y_train,y_test=y[train_index],y[test_index]
      model.fit(x_train, y_train,batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test,y_test),
                          callbacks=[callback_2])
      model = load_model(floder_path +str(subject)+'__'+str(model_file)+'.h5')
      print('Model evaluation ',model.evaluate(x_test,y_test))
      loss,accuracy,precision,recall_1=model.evaluate(x_test,y_test)
      print('kf_fold',i," ", 'loss',loss,'Accurary',accuracy,'Precision',precision,'Recall',recall_1)
      i=i+1
      accuracy_v.append(accuracy)
  return accuracy_v,print('avg', sum(accuracy_v)/10 )




def mark_img(dat):
  gram = GramianAngularField(image_size=16, method='summation')##change method='difference' for GramianAngularField difference
  gram_t = gram.fit_transform(dat.iloc[:,:-1])
  x = gram_t.reshape(gram_t.shape[0], 16, 16,1)
  y=dat['label']
  return x,y


def kf_flod_three(kf,x,y,floder_path,model_file,subject,epochs,batch_size):
    model1 = Sequential()
    model1.add(Conv2D(32, (2, 2), activation='relu', input_shape=(16, 16,1)))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Conv2D(64, (2, 2), activation='relu'))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Flatten())
    model1.add(Dense(32, activation='relu'))
    model1.add(Dense(8,activation='relu'))
    model1.add(Dense(3,activation='softmax'))
    model1.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy','Precision','Recall'])
   
    callback_2 = ModelCheckpoint(floder_path +str(subject)+'__'+str(model_file)+'.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


    cv = StratifiedKFold(n_splits=int(kf),shuffle=True,random_state=98)
    accuracy_v=[]
    y_true=[]
    y_predict = []
    pred_probs=[]
    tprs = []
    aucs = []
    fold_no=0
    i=1
    for train_index,test_index in cv.split(x,y):
        x_train,x_test=x[train_index],x[test_index]
        y_train,y_test=y[train_index],y[test_index]
        y_train = tf.keras.utils.to_categorical(y_train, num_classes = 3)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes = 3)


        model1.fit(x_train, y_train,batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test,y_test),
                            callbacks=[callback_2])
        model1 = load_model(floder_path +str(subject)+'__'+str(model_file)+'.h5')
        loss,accuracy,precision,recall_1=model1.evaluate(x_test,y_test)
        print('kf_fold',i," ", 'loss',loss,'Accurary',accuracy,'Precision',precision,'Recall',recall_1)
        i=i+1
        accuracy_v.append(accuracy)    
    return accuracy_v,print('avg', sum(accuracy_v)/10 )


