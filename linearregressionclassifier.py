import csv
import numpy as np
import sys
import getopt
import timeit
import random
from numpy import linalg as LA


filename = ""
datasize = 0
MAXITER = 500

bias = 1
etta = 0.1
w0 = np.zeros((1,5))
w1 = np.zeros((1,5))

start = timeit.default_timer() # start time of the program

# read arguments from command line
try:
    arguments, values = getopt.getopt(sys.argv[1:], 'hd:b:', ["help","dataset=","bias="])
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

# evaluate given options
for currentArgument, currentValue in arguments:
    if currentArgument in ("-h", "--help"):
        print ("Usage:")
        print("-v or --version to select [naive] or [pocket] perceptron")
        print("-d or --dataset to use [datasetname]")
        print("-h or --help to print this menu")
        print("-b or --bias to enter bias")
    elif currentArgument in ("-d", "--dataset"):
        print (("dataset filename: (%s)") % (currentValue))
        filename = currentValue
    elif currentArgument in ("-b", "--bias"):
        bias = float(currentValue)
        print("bias = ", bias)


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

def lin_reg(x,y):
    a = np.matmul(x.transpose(), x)
    # print("Linear regression, matrix a")
    # print(a)
    b = np.matmul(x.transpose(), y)
    # print("Linear regression, matrix b")
    # print(b)
    # w = np.matmul(np.linalg.inv(A), B)
    w_eig, v_eig = LA.eig(a)
    # print("W-eig, v-eig", w_eig, v_eig)
    d_plus = np.diag(1/w_eig)
    a_plus=np.matmul(v_eig, np.matmul(d_plus, np.linalg.inv(v_eig)))
    w = np.matmul(a_plus, b)
    # print(w)
    return w

# returns expected label for the vector (either 0 or 1) depends on which w vector it is closer to
def dist(w_0, w_1, x):
    a = np.dot(x, w_0)/np.linalg.norm(w_0)
    b = np.dot(x, w_1)/np.linalg.norm(w_1)
    if a < b:
        return 0
    else:
        return 1

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

x0_tmp = np.zeros((datasize,6))
x1_tmp = np.zeros((datasize,6))
y0 = np.zeros((datasize,1))

y1 = np.zeros((datasize,1))
x0_count = 0
x1_count = 0
w0[0] = bias
w1[0] = bias


class Var:
    size = 0
    r = -1
    ind = np.zeros((datasize, 1))

try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        k = 0
        for row in csv_reader:
            if k == 0:
                k += 1
            elif float(row[5]) > 1:
                x0_tmp[x0_count][0] = bias
                x0_tmp[x0_count][1] = float(row[0])
                x0_tmp[x0_count][2] = float(row[1])
                x0_tmp[x0_count][3] = float(row[2])
                x0_tmp[x0_count][4] = float(row[3])
                x0_tmp[x0_count][5] = float(row[4])
                x0_count += 1
            else:
                x1_tmp[x1_count][0] = bias
                x1_tmp[x1_count][1] = float(row[0])
                x1_tmp[x1_count][2] = float(row[1])
                x1_tmp[x1_count][3] = float(row[2])
                x1_tmp[x1_count][4] = float(row[3])
                x1_tmp[x1_count][5] = float(row[4])
                x1_count += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)



# print(x0_tmp[0:x0_count-1,:])
x0 = x0_tmp[0:x0_count,0:5].copy()
x1 = x1_tmp[0:x1_count,0:5].copy()
y0 = x0_tmp[0:x0_count,5:6].copy()
y1 = x1_tmp[0:x1_count,5:6].copy()
xbig = np.append(x0,x1,axis=0)
ybig = np.append(y0,y1,axis=0)

# print(xbig)

# w0 = lin_reg(xbig,ybig)
w0 = lin_reg(x0,y0)
w1 = lin_reg(x1,y1)
print(w0)
x0_miss = 0
x1_miss = 0

for j in range(x0_count):
    if dist(w0, w1, x0[j:(j+1), :]) != 1:
        x0_miss += 1

for j in range(x1_count):
    if dist(w0 , w1, x1[j:(j+1), :]) != 0:
        x1_miss += 1

print(x0_miss)
print(x1_miss)

# for j in x1:
#     if dist(w0, w1, x0[j]) != 1:
#         w1_miss += 1
#     if dist(w0, w1, x1[j]) != 0:
#         w0_miss += 1

# time of the program ends
stop = timeit.default_timer()

print('Time: ', stop - start)


