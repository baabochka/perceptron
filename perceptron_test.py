import numpy as np
import csv
from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))
with open('eggs.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print ', '.join(row)

labels = np.array([1, 0, 0, 0])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1])
perceptron.predict(inputs)
#=> 1
print("Hahah")
print(perceptron.predict(inputs))
inputs = np.array([0, 1])
perceptron.predict(inputs)
#=> 0
print("Hahah")
print(perceptron.predict(inputs))
