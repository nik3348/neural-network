import numpy as np
import pickle
from src.objects.NeuralNetwork import NeuralNetwork


MODEL = 'xor-new.nn'
try:
    # Deserialization
    with open("models/" + MODEL, "rb") as infile:
        nn = pickle.load(infile)
except:
    nn = NeuralNetwork([2, 2, 1])

data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float128)
result = np.array([[0], [1], [1], [0]], dtype=np.float128)
for x in range(10000):
    nn.train(data, result)

prediction = nn.predict(np.array([[0, 0]]))
print(prediction)

prediction = nn.predict(np.array([[1, 0]]))
print(prediction)

prediction = nn.predict(np.array([[0, 1]]))
print(prediction)

prediction = nn.predict(np.array([[1, 1]]))
print(prediction)

# Serialization
with open("models/" + MODEL, "wb") as outfile:
    pickle.dump(nn, outfile)

# nn.show()
