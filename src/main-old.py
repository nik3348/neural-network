import pickle
import numpy as np
import input_data as id
from src.objects.NeuralNetwork import NeuralNetwork


MODEL = 'sonar.nn'
EPOCHS = 10000
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
    array=np.zeros(2)
    array[int(ttarget[i])] = 1
    target[i]= array

for x in range(EPOCHS):
    nn.train(traw, target)

for i in range(len(praw)):
    prediction = nn.predict(np.array([praw[i]]))
    # print(prediction)
    print(np.argmax(prediction))
    print(ptarget[i])

if is_from_file:
    # Serialization
    with open("models/" + MODEL, "wb") as outfile:
        pickle.dump(nn, outfile)

# nn = NeuralNetwork([11, 11, 11, 10], True)
# target = np.zeros(shape=(len(ttarget), 10))
# for i in range(len(ttarget)):
#     # This is for classification
#     array=np.zeros(10)
#     array[int(ttarget[i])-1] = 1
#     target[i]= array

# nn.train(traw, target)

# for i in range(len(ptarget)):
#     prediction = nn.predict(np.array([praw[i]]).T)
#     print(np.argmax(prediction) + 1)
#     print(ptarget[i])

# prediction = nn.predict(np.array([praw[0]]).T)
# print(prediction)
# print(np.argmax(prediction) + 1)
# print(ptarget[0])

# print(id.extract_images('data/train-images-idx3-ubyte.gz')[0])
