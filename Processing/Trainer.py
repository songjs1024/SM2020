# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:57:48 2020

@author: junes
"""
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

with tf.device("/CPU:0"):
    data_path = "TrainingInfo.txt"
    info = pd.read_csv(data_path, header = None)
    raw_data = pd.read_csv(info.at[0,0].replace("\\","/"), header = None)
    raw_data.columns = ['value']
    cols = int(info.at[1,0])#x칼럼
    rows = int(info.at[2,0])#x로우
#x 데이터 셋 구성
    tempt_list = []
    for i in range(rows):
        tempt_list.append(list(raw_data['value'][i*cols:(i+1)*cols]))

#데이터셋 구성    
#y_type = info.at[4,0]#분류, 회귀(NN) 모델 분류
    x = np.array(tempt_list)
    y = np.loadtxt(info.at[3,0].replace("\\","/"))[:len(x)].reshape(-1, 1)
#option = pd.read_csv(AI_path)

    epochStr=info.at[8,0]
    if epochStr == '':
        nEpoch=100
    else:
        nEpoch=int(epochStr)

#assert(tf.config.list_physical_devices('GPU'))

#gpus = tf.config.experimental.list_physical_devices('/GPU:0')
## try:
  #  tf.config.experimental.set_memory_growth(gpus[0], True)
  #except RuntimeError as e:      
   # print(e)
#tf.keras.backend.clear_session()
#tf.config.optimizer.set_jit(False)
    optType=int(info.at[9,0])
    optLr=float(info.at[10,0])
    optDecay=float(info.at[11,0])
    optMomentum=float(info.at[12,0])
    optRho=float(info.at[13,0])
    optBeta1=float(info.at[14,0])
    optBeta2=float(info.at[15,0])
    optEpsilon=float(info.at[16,0])
    optOptioni=int(info.at[17,0])
    optOption=(optOptioni>0)
    if optType==0:
        optim = keras.optimizers.SGD(lr=optLr, momentum=optMomentum, decay = optDecay, nesterov=optOption)
    elif optType==1:
        if(optEpsilon<0.0):
            optim = keras.optimizers.RMSprop(lr=optLr, rho=optRho, epsilon=None, decay=optDecay)
        else:
            optim = keras.optimizers.RMSprop(lr=optLr, rho=optRho, epsilon=optEpsilon, decay=optDecay)
    elif optType==2:
        if(optEpsilon<0.0):
            optim = keras.optimizers.Adagrad(lr=optLr, epsilon=None, decay=optDecay)
        else:
            optim = keras.optimizers.Adagrad(lr=optLr, epsilon=optEpsilon, decay=optDecay)
    elif optType==3:
        if(optEpsilon<0.0):
            optim = keras.optimizers.Adadelta(lr=optLr, rho=optRho, epsilon=None, decay=optDecay)
        else:
            optim = keras.optimizers.Adadelta(lr=optLr, rho=optRho, epsilon=optEpsilon, decay=optDecay)
    elif optType==4:
        if(optEpsilon<0.0):
            optim = keras.optimizers.Adam(lr=optLr, beta_1=optBeta1, beta_2=optBeta2, epsilon=None, decay=optDecay, amsgrad=optOption)
        else:
            optim = keras.optimizers.Adam(lr=optLr, beta_1=optBeta1, beta_2=optBeta2, epsilon=optEpsilon, decay=optDecay, amsgrad=optOption)
    elif optType==5:
        if(optEpsilon<0.0):
            optim = keras.optimizers.Adamax(lr=optLr, beta_1=optBeta1, beta_2=optBeta2, epsilon=None, decay=optDecay)
        else:
            optim = keras.optimizers.Adamax(lr=optLr, beta_1=optBeta1, beta_2=optBeta2, epsilon=optEpsilon, decay=optDecay)
    elif optType==6:
        if(optEpsilon<0.0):
            optim = keras.optimizers.Nadam(lr=optLr, beta_1=optBeta1, beta_2=optBeta2, epsilon=None, schedule_decay=optDecay)
        else:
            optim = keras.optimizer.Nadam(lr=optLr, beta_1=optBeta1, beta_2=optBeta2, epsilon=optEpsilon, schedule_decay=optDecay)

    nLayer=int(info.at[18,0])
    layerList=[]
    for i in range(nLayer):
        nUnit = int(info.at[19+2*i,0])
        actStr=info.at[20+2*i,0]
        if i==0:
            layerList.append(keras.layers.Dense(nUnit, input_shape = (cols,), activation=actStr))
        elif i==nLayer-1:
            layerList.append(keras.layers.Dense(1))
        else:
            layerList.append(keras.layers.Dense(nUnit, activation=actStr))

    model =  keras.Sequential(layerList)
    #tf.config.optimizer.set_jit(True)
    model.compile(loss = 'mean_squared_error',
                  optimizer = optim,
                  metrics = ['mae','mse'])
    #callback = tf.keras.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
   # hist = model.fit(x,y, epochs = nEpoch, batch_size=200, callbacks = [callback]) 
    hist = model.fit(x,y, epochs = nEpoch) 

    #train image save
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
        
    loss_ax.plot(hist.history['mae'], 'y', label='train loss')
    acc_ax.plot(hist.history['mse'], 'b', label='train acc')
       
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
        
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    fig.savefig(info.at[5,0])

    numpy_loss_history = np.array(hist.history["mse"])
    np.savetxt(info.at[6,0],numpy_loss_history , delimiter=",")
    numpy_loss_history = np.array(hist.history["mae"])
    np.savetxt(info.at[7,0],numpy_loss_history  , delimiter=",") 
        
model.save(info.at[4,0])        
        











   
        


