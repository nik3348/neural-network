import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
from objects.NeuralNetwork import NeuralNetwork
from input_data import read_data_sets, plot_image


MODEL = 'mnist.nn'
EPOCHS = 3
is_from_file = True
NEW = not is_from_file

if is_from_file:
    try:
        # Deserialization
        with open("models/" + MODEL, "rb") as infile:
            nn = pickle.load(infile)
    except:
        NEW = True

if not is_from_file or NEW:
    nn = NeuralNetwork([784, 64, 10], 1e-4)

data_raw = read_data_sets('./data')
data_training = np.array(data_raw.__dict__['train'].__dict__['_images'], dtype=np.float128)
data_result = np.array(data_raw.__dict__['train'].__dict__['_labels'], dtype=np.float128)

# n = 6
# print(data_raw.__dict__['train'].__dict__['_labels'][n])
# plot_image([x.reshape(28,28) for x in data_training][n])

result = []
for x in range(len(data_result)):
    zeros = np.zeros(10)
    zeros[int(data_result[x])] = 1
    result.append(zeros)
result = np.array(result)

# x = []
# y = []
s1 = time.time()
for e in range(EPOCHS):
    nn.train(data_training, result)

    test_training = np.array(data_raw.__dict__['test'].__dict__['_images'], dtype=np.float128)
    test_result = np.array(data_raw.__dict__['test'].__dict__['_labels'], dtype=np.float128)

    prediction = nn.predict(data_raw.__dict__['test'].__dict__['_images'])
    acc = 0
    for i in range(len(prediction)):
        acc += (np.argmax(prediction[i]) == data_raw.__dict__['test'].__dict__['_labels'][i])
    print(acc * 100 /len(prediction))
    # x.append(e)
    # y.append(acc * 100 /len(prediction))

# plt.plot(x, y)
# plt.show()
s2 = time.time()
print(f's3 = {(s2 - s1)}')


if is_from_file:
    # Serialization
    with open("models/" + MODEL, "wb") as outfile:
        pickle.dump(nn, outfile)
