#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import pyaudio
import speech_data
import numpy
import os

batch=speech_data.wave_batch_generator(10000,target=speech_data.Target.digits)
X,Y=next(batch)

number_classes=10 # Digits

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y,n_epoch=10,show_metric=True,snapshot_step=100)
model.save("model.tfm")
# model.load("model.tfm")

baza_path = "data/spoken_numbers_pcm/tt/"
cnt = 0
sum = 0
for filename in os.listdir(baza_path):
    if filename.endswith(".wav"):
        demo=speech_data.load_wav_file(baza_path + filename)
        result=model.predict([demo])
        result=numpy.argmax(result)
        print("predicted speaker for %s : result = %d "%(filename, result))
        out = cluster_vector.cluster(baza_path_vect, os.listdir(baza_path_vect), filename[:-4])
        correct_cnt = 0
        for it in out:
            if it==''.join(filter(str.islower, filename[:-4])):
                correct_cnt += 1
        sum += (correct_cnt-1)/2
        cnt +=1
print(sum)
