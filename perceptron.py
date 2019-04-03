import csv
import numpy as np
import sys
import getopt
import timeit
import random

version = "naive"
filename = ""
datasize = 0
MAXITER = 500

bias = 1
etta = 0.1
w = np.zeros((1,6))
w_best = np.zeros((1,6))

start = timeit.default_timer() # start time of the program

# read arguments from command line
try:  
    arguments, values = getopt.getopt(sys.argv[1:], 'hv:d:b:n:m:', ["help","version=","dataset=","bias=","etta=","maxiter="])
except getopt.error as err:  
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

# evaluate given options
for currentArgument, currentValue in arguments:  
    if currentArgument in ("-v", "--version"):
        print (("program runs in %s mode") % (currentValue))
        version = currentValue
    elif currentArgument in ("-h", "--help"):
        print ("Usage:")
        print("-v or --version to select [naive] or [pocket] perceptron")
        print("-d or --dataset to use [datasetname]")
        print("-b or --bias to enter bias")
        print("-n or --etta to enter etta")
        print("-m or --MAXITER to set maximum number of iterations")
        print("-h or --help to print this menu") 
    elif currentArgument in ("-d", "--dataset"):
        print (("dataset filename: (%s)") % (currentValue))
        filename = currentValue
    elif currentArgument in ("-b", "--bias"):
        bias = float(currentValue)
        print("bias = ", bias)
    elif currentArgument in ("-n", "--etta"):
        etta = float(currentValue)
        print("etta = ", etta)
    elif currentArgument in ("-m", "--maxiter"):
        MAXITER = float(currentValue)
        print("MAXITER = ", MAXITER)

def loss_fn(test_w):
    count = 0
    for i in range(datasize):
        value = np.dot(test_w, x[i])
        if value >= 0:
            value = 1
        else:
            value = -1
        if value != y[i]:
            count = count + 1
    return count / datasize*1.0

def rand_order():
    r = random.randint(0, datasize-1)
    while(True):
        if (Var.ind[r] == 0):
            Var.ind[r] = 1
            Var.size = Var.size + 1
            return r
        else:
            if Var.size == datasize:
                Var.size = 0
                Var.ind = np.zeros((datasize, 1))
                return -1
            r = random.randint(0, datasize - 1)
try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
        datasize = line_count - 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)

x = np.zeros((datasize,6))
y = np.zeros((datasize,1))
w[0] = bias
w_best[0] = bias

class Var:
    size = 0
    r = -1
    ind = np.zeros((datasize, 1))

try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                x[line_count - 1][0] = 1
                x[line_count - 1][1] = float(row[0])
                x[line_count - 1][2] = float(row[1])
                x[line_count - 1][3] = float(row[2])
                x[line_count - 1][4] = float(row[3])
                x[line_count - 1][5] = float(row[4])
                if float(row[5]) < 1:
                    y[line_count - 1] = -1
                else:
                    y[line_count - 1] = float(row[5])
                line_count += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)


if version == 'naive':
    count = 1

    while(count < MAXITER):
        while(True):
            i = rand_order()
            if i == -1:
                break
            res = y[i] * np.dot(w,x[i])
            if res <= 0:
                w = w + y[i] * x[i]

        count += 1

        if count % 100 == 0:
            print(f"Iteration {count}: LOSS: {loss_fn(w)}")
    print(f"Empirical risk minimization: {loss_fn(w)}")
    print(w)

elif version == 'pocket':
    count = 1
    best_run = 0
    cur_run = 0

    while (count < MAXITER):

        while (True):
            i = rand_order()
            if i == -1:
                break
            res = y[i] * np.dot(w, x[i])
            if res > 0:
                cur_run += 1
                if cur_run > best_run:
                    best_run = cur_run
                    w_best = w
            else:
                w = w + etta * x[i] * y[i]
                cur_run = 0

        count += 1

        if count % 100 == 0:
            print(f"Iteration {count}: LOSS: {loss_fn(w)}")
            print(f"Iteration {count}: BEST LOSS: {loss_fn(w_best)}")
    print(f"Empirical risk minimization: {loss_fn(w_best)}")
    print(w_best)

# time of the program ends
stop = timeit.default_timer()

print('Time: ', stop - start)