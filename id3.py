import csv
import numpy as np
import sys
import getopt
import timeit
from numpy import linalg as LA


filename = ""
datasize = 0
MAXITER = 500

bias = -0.09
etta = 0.1
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



# open a file to calculate datasize for each label
try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        size0 = 0
        size1 = 0
        k = 0
        for row in csv_reader:
            if k == 0:
                k += 1
            elif float(row[5]) < 1:
                size0 += 1
            else:
                size1 += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)


# initialize vector of features and label with zeros
x0 = np.zeros((size0,5))
x1 = np.zeros((size1,5))
y0 = np.zeros((size0,1))
y1 = np.zeros((size1,1))

#open the file and fill the data into vectors
try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        k = 0
        x0_count = 0
        x1_count = 0
        for row in csv_reader:
            # print(row)
            if k == 0:
                k += 1
            elif float(row[5]) < 1:
                # adding extra dimension to handle the bias and filling x_i and y_i with data
                x0[x0_count][0] = 1
                x0[x0_count][1] = float(row[0])
                x0[x0_count][2] = float(row[1])
                x0[x0_count][3] = float(row[2])
                x0[x0_count][4] = float(row[3])
                y0[x0_count][0] = float(row[4])
                x0_count += 1
            else:
                x1[x1_count][0] = 1
                x1[x1_count][1] = float(row[0])
                x1[x1_count][2] = float(row[1])
                x1[x1_count][3] = float(row[2])
                x1[x1_count][4] = float(row[3])
                y1[x1_count][0] = float(row[4])
                x1_count += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)


# find weight vectors for each set
w0 = lin_reg(x0,y0)
w1 = lin_reg(x1,y1)

x0_miss = 0
x1_miss = 0

# find the number of misses for the whole set
for j in range(x0_count):
    if dist(w0, w1, x0[j:(j+1), :]) != 0:
        x0_miss += 1

for j in range(x1_count):
    if dist(w0 , w1, x1[j:(j+1), :]) != 1:
        x1_miss += 1

print("Number of mislabelled vectors in first(label-0) set is: ", x0_miss)
print("Number of mislabelled vectors in second(label-1) set is: ", x1_miss)
print("Classifier accuracy: ", round((1-(x0_miss+x1_miss)/(x0_count+x1_count))*100,2), "%")

# for j in x1:
#     if dist(w0, w1, x0[j]) != 1:
#         w1_miss += 1
#     if dist(w0, w1, x1[j]) != 0:
#         w0_miss += 1

# time of the program ends
stop = timeit.default_timer()

print('Time: ', stop - start)


