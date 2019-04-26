import csv
import numpy as np
import pandas as pd
import sys
import getopt
import timeit
import math
from numpy import linalg as LA


filename = ""
datasize = 0
have_ig = 1

train = 0
reshuffle = 0

start = timeit.default_timer() # start time of the program

# read arguments from command line
try:
    arguments, values = getopt.getopt(sys.argv[1:], 'hd:ntr', ["help","dataset=","train","test","reshuffle"])
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

# evaluate given options
for currentArgument, currentValue in arguments:
    if currentArgument in ("-h", "--help"):
        print ("Usage:")
        print("-h or --help to print this menu")
        print("-d or --dataset to use [datasetname]")
        print("-n or --train to train algorithm on a 60% of a dataset")
        print("-t or --test to test algorithm on a 40% of a dataset")

    elif currentArgument in ("-d", "--dataset"):
        print (("dataset filename: (%s)") % (currentValue))
        filename = currentValue
    elif currentArgument in ("-n", "--train"):
        train = 1
        print("Let's train the algorithm")
    elif currentArgument in ("-t", "--test"):
        train = 0
        print("Let's test the algorithm")
    elif currentArgument in ("-r", "--reshuffle"):
        reshuffle = 1
        print("Let's reshuffle the dataset")

# function to calculate information gain


# convert string values in columns to ints
def stringToInt(input, i_type):
    if input == '' and i_type == 'age':
        output = 26
    elif input == '':
        output = 0
    elif input == 'C':
        output = 0
    elif input == 'Q':
        output = 1
    elif input == 'S':
        output = 2
    elif input == 'male':
        output = 1
    elif input == 'female':
        output = 0
    else:
        output = float(input)
    return output


def calculate_entropy(probs):
    entr = 0
    for p in probs:
        entr += p * math.log2(p)
    return -1 * entr


# open a file to calculate datasize
try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        size = 0
        for row in csv_reader:
            size += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)


# initialize vector of features and label with zeros

datasize = size - 1
size60 = int(datasize * 0.6)
size40 = datasize - size60
x60 = np.zeros((size60,8))
y60 = np.zeros((size60,1))
x40 = np.zeros((size40,8))
y40 = np.zeros((size40,1))
z = []
#open the file and fill the data into vectors splitting it into 60% and 40%
try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        k = 0
        index = 0
        index40 = 0
        for row in csv_reader:
            if k == 0:
                k += 1
            elif k <= size60:
                y60[index][0] = float(row[1])         # Survived
                x60[index][0] = float(row[2])         # Pclass
                x60[index][1] = stringToInt(row[4],'')   # Sex (male = 1, female = 0)
                x60[index][2] = stringToInt(row[5],'age')   # Age (blank = 0 for now)
                x60[index][3] = float(row[6])         # SibSp
                x60[index][4] = float(row[7])         # Parch
                x60[index][5] = float(row[9])         # Fare
                x60[index][6] = stringToInt(row[11], '')  # Embarked (C = 0, Q = 1, S = 2)
                # x[index][7] = float(row[0])       # Passenger number
                # z.append(row[3])                  # Passenger Name
                index += 1
                k += 1
            else:
                y40[index40][0] = float(row[1])  # Survived
                x40[index40][0] = float(row[2])  # Pclass
                x40[index40][1] = stringToInt(row[4], '')  # Sex (male = 1, female = 0)
                x40[index40][2] = stringToInt(row[5], 'age')  # Age (blank = 0 for now)
                x40[index40][3] = float(row[6])  # SibSp
                x40[index40][4] = float(row[7])  # Parch
                x40[index40][5] = float(row[9])  # Fare
                x40[index40][6] = stringToInt(row[11], '')  # Embarked (C = 0, Q = 1, S = 2)
                # x[index][7] = float(row[0])       # Passenger number
                # z.append(row[3])                  # Passenger Name
                index40 += 1
                k += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)

# stop1 = 0
# print("Pclass  Sex  Age  SibSp  Parch  Fare Embarked Survived Name")
# for i in range(size - 1):
#     print(x40[i], y40[i], z[i])
#     stop1 += 1
#     if stop1 == 10:
#         break


stop = timeit.default_timer()

print('Time: ', stop - start)
print('Size: ', size)

entropy_surv = 0
surv = 0
surv_not = 0
prob_surv = []
prob_surv1 = []
c1 = 0
c2 = 0
c3 = 0
y1c1 = 0
y0c1 = 0
y1c2 = 0
y0c2 = 0
y1c3 = 0
y0c3 = 0

# function for finding information gain for a certain feature
def find_ig(feature_vector, label_vector, categories, separators):
    surv_not = 0
    surv = 0
    flag = 0
    i = 0
    feature = np.zeros((len(categories),1))
    label = np.zeros((len(categories),2))
    total_category = 0
    cond_entropy = 0
    total_label_for_feature = []
    for l in range(len(label_vector[0])):
        if label_vector[0][l] == 0:
            surv_not += 1
        else:
            surv += 1
    target_entr = calculate_entropy([surv / len(label_vector[0]), surv_not / len(label_vector[0])])
    if len(separators) == 0:
        for f in range(len(feature_vector)):
            i = 0
            while i < len(categories):
                # print("feature_vector[f]  ",feature_vector[f], " categories[i] ", categories[i], " label_vector[f] ", label_vector[0][f], " i=", i)
                if feature_vector[f] == categories[i]:
                    feature[i] += 1
                    if label_vector[0][f] == 0:
                        label[i][0] += 1
                    else:
                        label[i][1] += 1
                i += 1
    else:
        for f in range(len(feature_vector)):
            i = 1
            while i < len(separators):
                # print("feature_vector[f]  ",feature_vector[f], " separators[i] ", separators[i], " label_vector[f] ", label_vector[0][f], " i=", i)
                if separators[i-1] < feature_vector[f] <= separators[i]:
                    feature[i-1] += 1
                    if label_vector[0][f] == 0:
                        label[i-1][0] += 1
                    else:
                        label[i-1][1] += 1
                i += 1

    for i in range(len(categories)):
        total_category += feature[i]

    for i in range(len(categories)):
        total_label_for_feature.append(label[i][0] + label[i][1])

    for i in range(len(categories)):
        cond_entropy += ((label[i][0] + label[i][1]) / total_category) * calculate_entropy([label[i][0]/total_label_for_feature[i], label[i][1]/total_label_for_feature[i]])

    return target_entr - cond_entropy


a = find_ig(x60[:,2], np.transpose(y60), [1,2,3], [0,30,60,90])
print("Please, find an IG correct: ", a)
b = find_ig(x60[:,0], np.transpose(y60), [1,2,3], [])
print("Please, find an IG correct: ", b)

# print("LEN: ", len(label_vector[0]))
# print("SURV: ", surv, " SURV_NOT: ", surv_not, " LEN: ", len(label_vector[0]))
# print("surv/len(label_vector[0]) = ", surv / len(label_vector[0]), " surv_not/len(label_vector[0]) = ",
#       surv_not / len(label_vector[0]))
#
# print("TARG_ENTR: ", target_entr)


