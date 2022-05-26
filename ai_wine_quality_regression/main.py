import numpy as np
from ai_wine_quality_regression.objects.NeuralNetwork import NeuralNetwork


# Prediction data
pdata = np.genfromtxt('data/test.csv', delimiter=';', skip_header=1)
praw = pdata[:, :-1].copy()
ptarget = pdata[:, -1].copy()

# Training data
tdata = np.genfromtxt('data/winequality-white.csv', delimiter=';', skip_header=1)
traw = tdata[:, :-1].copy()
ttarget = tdata[:, -1].copy()

nn = NeuralNetwork([11, 11, 11, 10])

for i in range(len(traw)):
    array=np.zeros(10)
    array[int(ttarget[i])-1]=1
    nn.train(np.array([traw[i]]).T, array.reshape(10, -1))

prediction = nn.predict(np.array([praw[1]]).T)
print(prediction)
print(ptarget[1])
