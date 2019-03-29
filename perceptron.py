import csv
import numpy as np
import sys
import getopt


version = "naive"
filename = ""
datasize = 0
MAXITER = 30

bias = 0.01
eps = 0.001

x = np.zeros((569,5))
y = np.zeros((569,1))
w = np.zeros((1,5))
w_pok = np.zeros((1,5))

# read arguments from command line
try:  
    arguments, values = getopt.getopt(sys.argv[1:], 'hv:d:', ["help","version=","dataset="])
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
        print("-h or --help to print this menu") 
    elif currentArgument in ("-d", "--dataset"):
        print (("dataset filename: (%s)") % (currentValue))
        filename = currentValue


def loss_fn():
    count = 0
    for i in range(datasize):
        value = np.dot(w, x[i]) + bias
        if value >= 0:
            value = 1
        else:
            value = -1
        if value != y[i]:
            count = count + 1
    return count / datasize*1.0
try:
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                x[line_count - 1][0] = float(row[0])
                x[line_count - 1][1] = float(row[1])
                x[line_count - 1][2] = float(row[2])
                x[line_count - 1][3] = float(row[3])
                x[line_count - 1][4] = float(row[4])
                if float(row[5]) < 1:
                    y[line_count - 1] = -1
                else:
                    y[line_count - 1] = float(row[5])
                line_count += 1
        datasize = line_count - 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)

updated = True
count = 1
min_loss = 1

if version == 'naive':
    while(updated):
        updated = False
        prev_loss = loss_fn()
        print(f"Iteration {count}: LOSS: {loss_fn()}")
        for i in range(datasize):
            res = y[i] * np.dot(w,x[i])
            if res <= 0:
                w = w + y[i] * x[i]
                updated = True
                break
        count = count + 1
        new_loss = loss_fn()
        loss_diff = new_loss - prev_loss
        if loss_diff < 0:
            loss_diff = loss_diff * (-1)
        if min_loss > new_loss and count < datasize*MAXITER*0.6: #10000
            min_loss = new_loss
        if min_loss < 0.05:
            print(f"Iteration {count}: LOSS: {new_loss}")
            break;
        if loss_diff < eps and count > datasize*MAXITER*0.3: #5000
            print(f"Iteration {count}: LOSS: {new_loss}")
            break;
        if count >= datasize*MAXITER*0.6 and new_loss < min_loss: #10000
            print(f"Iteration {count}: LOSS: {new_loss}")
            min_loss = new_loss
            break;
        if count > datasize*MAXITER: #15000
            print(f"Iteration {count}: LOSS: {new_loss}")
            break;
    print(f"Min loss: {min_loss}")
    print(w)
elif version == 'pocket':
    while (updated):
        updated = False
        prev_loss = loss_fn()
        print(f"Iteration {count}: LOSS: {loss_fn()}")
        for i in range(datasize):
            res = y[i] * np.dot(w, x[i])
            if res <= 0:
                w = w + y[i] * x[i]
                updated = True
                break
        count = count + 1
        new_loss = loss_fn()
        loss_diff = new_loss - prev_loss
        if loss_diff < 0:
            loss_diff = loss_diff * (-1)
        if min_loss > new_loss and count < datasize * MAXITER * 0.6:  # 10000
            min_loss = new_loss
        if min_loss < 0.05:
            print(f"Iteration {count}: LOSS: {new_loss}")
            break;
        if loss_diff < eps and count > datasize * MAXITER * 0.3:  # 5000
            print(f"Iteration {count}: LOSS: {new_loss}")
            break;
        if count >= datasize * MAXITER * 0.6 and new_loss < min_loss:  # 10000
            print(f"Iteration {count}: LOSS: {new_loss}")
            min_loss = new_loss
            break;
        if count > datasize * MAXITER:  # 15000
            print(f"Iteration {count}: LOSS: {new_loss}")
            break;
    print(f"Min loss: {min_loss}")
    print(w)
