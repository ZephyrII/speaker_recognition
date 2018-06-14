#!/usr/bin/env python
#!/usr/bin/env python
#!/usr/bin/python
import os
import numpy as np
import tensorflow as tf
import tflearn
import librosa
import cluster_vector

import layer
import speech_data
from speech_data import Source,Target

batch_size = 64
height=20 # mfcc features
width=240 # (max) length of utterance
classes=22

batch = word_batch = speech_data.mfcc_batch_generator(batch_size, source=Source.DIGIT_WAVES, target=Target.speaker)
X, Y = next(batch)
print("batch shape " + str(np.array(X).shape))

speakers = speech_data.get_speakers()
shape=[-1, height, width, 1]
net = tflearn.input_data(shape=[None, 20, 240])
net = tflearn.conv_1d(net, 100, 16)
net = tflearn.relu(net)
fc2_dims = 64
fc2 = tflearn.fully_connected(net, fc2_dims, name='fc2')

net = tflearn.fully_connected(fc2, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
# model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
# model.save("model.tfm")
model.load("0.8 conv100,16 relu fc64/model.tfm")

m2 = tflearn.DNN(fc2, session=model.session)
cnt = 0
sum = 0
baza_path = "data/baza_glosow/"
baza_path_vect = "data/baza_glosow/vect/"


# Feature extraction phase
for filename in os.listdir(baza_path):
    if filename.endswith(".wav"):
        wave, sr = librosa.load(baza_path+filename, mono=True)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc=np.pad(mfcc,((0,0),(0,240-len(mfcc[0]))), mode='constant', constant_values=0)
        vect=m2.predict([np.array(mfcc)])
        sq_vect = np.squeeze(vect)
        np.savetxt(baza_path_vect+filename+".npz", sq_vect, delimiter=',')

# Verification phase
for filename in os.listdir(baza_path):
    if filename.endswith(".wav"):
        out = cluster_vector.cluster(baza_path_vect, os.listdir(baza_path_vect), filename[:-4],fc2_dims)
        correct_cnt = 0
        print("result = %s "%(out))
        if out[0]==out[1]:
            sum+=1
        cnt+=1
print(sum/cnt)
