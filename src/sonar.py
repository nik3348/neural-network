import pickle
import numpy as np
import input_data as id
from objects.NeuralNetwork import NeuralNetwork


MODEL = 'sonar.nn'
EPOCHS = 2500
is_from_file = True
NEW = not is_from_file

# Training data
tdata = np.genfromtxt('data/sonar-test.csv', delimiter=',')
traw = tdata[:, :-1].copy()
ttarget = tdata[:, -1].copy()
ttarget = np.reshape(ttarget, (-1, 1))

# Prediction data
pdata = np.genfromtxt('data/sonar-test.csv', delimiter=',')
praw = pdata[:, :-1].copy()
ptarget = pdata[:, -1].copy()

if is_from_file:
    try:
        # Deserialization
        with open("models/" + MODEL, "rb") as infile:
            nn = pickle.load(infile)
    except:
        NEW = True

if not is_from_file or NEW:
    nn = NeuralNetwork([60, 10, 10, 2])

target = np.zeros(shape=(len(ttarget), 2))
for i in range(len(ttarget)):
    # This is for classification
    array = np.zeros(2)
    array[int(ttarget[i])] = 1
    target[i] = array

for x in range(EPOCHS):
    nn.train(traw, target)

accuracy = 0
for i in range(len(praw)):
    prediction = nn.predict(np.array([praw[i]], dtype=np.float128))
    accuracy += np.argmax(prediction) == ptarget[i]
print((accuracy/len(praw))*100)

if is_from_file:
    # Serialization
    with open("models/" + MODEL, "wb") as outfile:
        pickle.dump(nn, outfile)
