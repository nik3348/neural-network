import numpy as np
import pickle
from objects.NeuralNetwork import NeuralNetwork


MODEL = 'xor.nn'
EPOCHS = 10000
is_from_file = False
NEW = not is_from_file

if is_from_file:
    try:
        # Deserialization
        with open("models/" + MODEL, "rb") as infile:
            nn = pickle.load(infile)
    except:
        NEW = True

if not is_from_file or NEW:
    nn = NeuralNetwork([2, 2, 1], 1e-1, True)

data = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float128)
result = np.array([0, 1, 0, 1], dtype=np.float128)
for x in range(EPOCHS):
    nn.train(data, result)

prediction = nn.predict(np.array([[0, 0]], dtype=np.float128))
print(prediction)

prediction = nn.predict(np.array([[1, 0]], dtype=np.float128))
print(prediction)

prediction = nn.predict(np.array([[0, 1]], dtype=np.float128))
print(prediction)

prediction = nn.predict(np.array([[1, 1]], dtype=np.float128))
print(prediction)

if is_from_file:
    # Serialization
    with open("models/" + MODEL, "wb") as outfile:
        pickle.dump(nn, outfile)
