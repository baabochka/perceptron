import csv
import numpy as np
import sys
import getopt
import timeit
import math

class Tree(object):
    def __init__(self):
        self.left = None
        self.child = []
        self.data = []
        self.name = ""

    def createChildren(self, amount):
        for i in range(0, amount):
            self.child.append(Tree())

    def setChildrenValues(self, list):
        for i in range(0, len(list)):
            self.data.append(list[i])

    def setName(self, name):
        self.name = name

    def setData(self, data):
        self.data = data


derevo = Tree()

filename = ""
datasize = 0
have_ig = 1
tree_depth = 6
reshuffle = 0
sep_age = [0, 15, 25, 40, 60, 100]
sep_fare = [-1, 20, 100, 550]
start = timeit.default_timer() # start time of the program

# read arguments from command line
try:
    arguments, values = getopt.getopt(sys.argv[1:], 'hd:t:r', ["help","dataset=","tree","reshuffle"])
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
        print("-t or --tree to set decision tree depth")

    elif currentArgument in ("-d", "--dataset"):
        print (("dataset filename: (%s)") % (currentValue))
        filename = currentValue
    elif currentArgument in ("-t", "--tree"):
        tree_depth = int(currentValue)
        print("DEPTH set to ", tree_depth)
    elif currentArgument in ("-r", "--reshuffle"):
        reshuffle = 1
        print("Let's reshuffle the dataset")



# convert string values in columns to ints
def stringToInt(input, i_type):
    output = -1
    if i_type == 'age':
        if input == '':
            input = 26
        for i in range(1, len(sep_age)+1):
            if sep_age[i-1] < float(input) <= sep_age[i]:
                output = round((sep_age[i] + sep_age[i-1])/2)
    elif i_type == 'fare':
        for i in range(1, len(sep_fare)+1):
            if sep_fare[i - 1] < float(input) <= sep_fare[i]:
                output = round((sep_fare[i] - sep_fare[i - 1]) / 2)
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


def calculate_entropy(probabilities):
    entr = 0
    for p in probabilities:
        if p != 0:
            entr += p * math.log2(p)
    return -1 * entr

# calculate size of each feature set and corresponding labels
def calculate_size_split(feature_vector, label_vector):
    c, s = extract_categories(feature_vector)
    if len(s) == 0:
        feature = np.zeros((len(c),1))
    else:
        feature = np.zeros((len(s), 1))
    label = np.zeros((len(c),2))
    if len(s) == 0:
        for f in range(len(feature_vector)):
            i = 0
            while i < len(c):
                if feature_vector[f] == c[i]:
                    feature[i] += 1
                    if label_vector[0][f] == 0:
                        label[i][0] += 1
                    else:
                        label[i][1] += 1
                i += 1
    else:
        for f in range(len(feature_vector)):
            i = 1
            while i < len(s):
                if s[i-1] < feature_vector[f] <= s[i]:
                    feature[i-1] += 1
                    if label_vector[0][f] == 0:
                        label[i-1][0] += 1
                    else:
                        label[i-1][1] += 1
                i += 1

    return feature, label

def survival_proportions(label_vector):
    surv = [0,0] # [not survived, survived]

    for l in range(len(label_vector[0])):
        if label_vector[0][l] == 0:
            surv[0] += 1
        else:
            surv[1] += 1
    return surv

def majorLabel(labels):
    l1 = 0
    l0 = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            l0 += 1
        else:
            l1 += 1
    if l0 > l1:
        return 0
    else:
        return 1




def dT(data, labels, column, depth):
    l_set = []
    cat = []
    split = 0
    for i in range(len(labels)):
        if labels[i] not in l_set:
            l_set.append(labels[i])
    if len(l_set) == 1:
        # print("We reached the leaf at category ", column)
        # print(majorLabel(labels))
        der = Tree()
        der.setData((majorLabel(labels)))
        # print("LEAF: ", der.data)
        return der
    if depth == tree_depth:
        der = Tree()
        der.setData((majorLabel(labels)))
        # print("LEAF: ", der.data)
        return der

    # print("LEVELS LEN: ", len(levels))
    # print("DEPTH ", depth)
    #
    #
    # print("data size: ", len(data))
    max_inf_gain = -1
    feature_index = -1
    for i in range(len(data[0])):
        categories, separators = extract_categories(data[:,i])
        c = len(categories)
        tmp_gain = find_ig(data[:,i], np.transpose(labels), categories, separators)
        # print("IG on feature ", i, " is equal to ", tmp_gain)
        if tmp_gain > max_inf_gain:
            max_inf_gain = tmp_gain
            feature_index = i
            split = c
            cat = categories
    # print("Here is the max info gain over all the features", max_inf_gain, " on index ", feature_index)

    splitted_data, splitted_label = split_data(data, labels, feature_index)
    der = Tree()
    der.setData((cat, feature_index))
    der.createChildren(split)
    for i in range(split):
        der.child[i] = dT(splitted_data[i], splitted_label[i], feature_index, depth + 1)
    return der


# function to split dataset by category feat
def split_data(data, labels, col):
    c,s = extract_categories(data[:,col])
    feature, label = calculate_size_split(data[:,col], np.transpose(labels))
    data_cat = []
    label_cat = []
    index_cat = []
    index_cat_counter = []
    cat_number = 0
    if len(s) == 0:
        cat_number = len(c)
    else:
        cat_number = len(s) - 1

    for m in range(cat_number):
        data_cat.append(np.zeros((int(feature[m][0]),7)))
        label_cat.append(np.zeros((int(feature[m][0]),1)))
        index_cat.append(int(feature[m][0]))
        index_cat_counter.append(0)

    if len(s) == 0:
        ind = 0
        for row in data:
            i = 0
            while i < len(c):
                if row[col] == c[i]:
                    data_cat[i][index_cat_counter[i]] = row
                    label_cat[i][index_cat_counter[i]] = labels[ind]
                    index_cat_counter[i] += 1
                i += 1
            ind += 1
    else:
        ind = 0
        for row in data:
            i = 1

            while i < len(s):
                if s[i-1] < row[col] <= s[i]:
                    data_cat[i-1][index_cat_counter[i-1]] = row
                    label_cat[i-1][index_cat_counter[i-1]] = labels[ind]
                    index_cat_counter[i-1] += 1
                i += 1
            ind += 1
    return data_cat, label_cat



# function for finding information gain for a certain feature
def find_ig(feature_vector, label_vector, categories, separators):

    total_category = 0
    cond_entropy = 0
    total_label_for_feature = []
    surv = survival_proportions(label_vector)

    target_entr = calculate_entropy([surv[1] / len(label_vector[0]), surv[0] / len(label_vector[0])])

    feature, label = calculate_size_split(feature_vector, label_vector)

    if len(separators) == 0:
        cat_number = len(categories)
    else:
        cat_number = len(separators) - 1

    for i in range(cat_number):
        total_category += feature[i]
    for i in range(cat_number):
        total_label_for_feature.append(label[i][0] + label[i][1])
    for i in range(cat_number):
        cond_entropy += ((label[i][0] + label[i][1]) / total_category) * calculate_entropy([label[i][0]/total_label_for_feature[i], label[i][1]/total_label_for_feature[i]])

    return target_entr - cond_entropy

# find distinct categories among the column (and separators if data is not discrete
def extract_categories(feature_vector):
    categories = []
    separators = []
    for i in range(len(feature_vector)):
        if feature_vector[i] not in categories:
            categories.append(feature_vector[i])
    categories.sort()
    if len(categories) > 8:
        separators.extend([np.percentile(feature_vector, 0)-0.5, np.percentile(feature_vector, 25), np.percentile(feature_vector, 50), np.percentile(feature_vector, 75), np.percentile(feature_vector, 100)])
    return categories, separators





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
x60 = np.zeros((size60,7))
y60 = np.zeros((size60,1))
x40 = np.zeros((size40,7))
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
                x60[index][5] = stringToInt(float(row[9]), 'fare')         # Fare
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
                x40[index40][5] = stringToInt(float(row[9]), 'fare')    # Fare
                x40[index40][6] = stringToInt(row[11], '')  # Embarked (C = 0, Q = 1, S = 2)
                # x[index][7] = float(row[0])       # Passenger number
                # z.append(row[3])                  # Passenger Name
                index40 += 1
                k += 1
except IOError:
    print('An error occured trying to read the file.')
    sys.exit(-1)


stop = timeit.default_timer()

print('Time: ', stop - start)
print('Size: ', size)




# root = Tree()
# root.setData(["root"])


# root1 = Tree()
# root1.setData(["root"])
# root1.createChildren(3)
# root1.setChildrenValues([5, 6, 7])
#
# root1.child[0].createChildren(2)
# root1.child[0].setChildrenValues([1, 2])
#
# root1.child[1].createChildren(1)
# root1.child[1].setChildrenValues([8])
#
# root1.child[2].createChildren(2)
# root1.child[2].setChildrenValues([4, 7])


# # print some values in the tree
# printTree(root1, root1.data, 0)
# print(root.child[0].data[0])

print("---------")
myTree = dT(x60, y60,0,0)

class Var():
    correct = 0
    wrong = 0


def test_data(node, test_vector, test_label):
    # print("SIZE: ", len(test_vector))
    # print("NODE = 1 ", node.data == 1)
    if node.data == 1 or node.data == 0:
        # print("NODE DATA: ", node.data)
        if node.data == test_label:
            Var.correct += 1
        else:
            Var.wrong += 1
        return node.data
    t = node.data[1]
    # print("len-node-data ", len(node.data[0]))
    for j in range(len(node.data[0])):
        if test_row[t] == node.data[0][j]:
            test_data(node.child[j], test_vector,test_label)


test_index = 0

for test_row in x40:
    tmp = test_data(myTree, test_row, y40[test_index])
    test_index += 1
print("WRONG test: ", Var.wrong, "CORRECT test: ", Var.correct)

test1_index = 0
Var.correct = 0
Var.wrong = 0
for test1_row in x60:
    tmp = test_data(myTree, test1_row, y60[test1_index])
    test_index += 1
print("WRONG train: ", Var.wrong, "CORRECT train: ", Var.correct)

# dT(x60, y60,0,0)
#
# for j in range(len(levels)):
#     print(j, "  ", levels)
    # list_of_splits.append(levels[j])

# print("LIST OF SPLITS: ", list_of_splits)
# printTree(root, root.data, 0)
print("---------")
print("TEST DATA")
# print(test_data(myTree))
print("---------")
print("---------")
# d,l = split_data(x60,y60,5)
# print(d[0][0], d[0][1])
# print(np.transpose(l[0]))

# print(np.transpose(l[1]))
# print(np.transpose(l[2]))

# print(extract_categories(x60[:,1]))

# print("D0 = ", d[0])


# print("y60 ", np.transpose(y60))
# print("L0 = ", np.transpose(l[0]))
# d1,l1 = split_data(d[0],l[0],1)
# print(d1[0][0],d1[1][0])
# print("LEN: ", len(label_vector[0]))
# print("SURV: ", surv, " SURV_NOT: ", surv_not, " LEN: ", len(label_vector[0]))
# print("surv/len(label_vector[0]) = ", surv / len(label_vector[0]), " surv_not/len(label_vector[0]) = ",
#       surv_not / len(label_vector[0]))
#
# print("TARG_ENTR: ", target_entr)


