#!/usr/bin/env python
#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os
import tflearn
import speech_data as data
import cluster_vector
import tensorflow as tf
import numpy as np

print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
	quit() # why? works on Mac?

number_classes=10

batch=data.wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.digits)
X,Y=next(batch)


# Classification
net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
fc1 = tflearn.fully_connected(net, 32, name='fc1')
do = tflearn.dropout(fc1, 0.5)
fc2 = tflearn.fully_connected(do, 16, name='fc2')
regression = tflearn.fully_connected(fc2, number_classes, activation='softmax')
net = tflearn.regression(regression, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(regression)
model.fit(X, Y, n_epoch=200, show_metric=True, snapshot_step=100)
model.save("model.tfl")
# model.load("model.tfl")
m2 = tflearn.DNN(fc2, session=model.session)

baza_path = "data/baza_glosow/"
baza_path_vect = "data/baza_glosow/vect/"


# Feature extraction phase
for filename in os.listdir(baza_path):
    if filename.endswith(".wav"):
        demo=data.load_wav_file(baza_path+filename)
        vect = m2.predict([demo])
        sq_vect = np.squeeze(vect)
        np.savetxt(baza_path_vect+filename+".npz", sq_vect, delimiter=',')

# Verification phase
cnt = 0
sum = 0
for filename in os.listdir(baza_path):
    if filename.endswith(".wav"):
        out = cluster_vector.cluster(baza_path_vect, os.listdir(baza_path_vect), filename[:-4])
        print(filename)
        print(out)
        correct_cnt = 0
        for it in out:
            if it==''.join(filter(str.islower, filename[:-4])):
                correct_cnt += 1
        sum += (correct_cnt-1)/2
        cnt +=1
print(sum/cnt)

