import numpy as np
import csv
import pandas as pd
from math import log
import sys
myname = "Vivek John George"

# Process Column Names
Attribute_Names = 'fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality'.split(',')
Attribute_Dict ={}
counter = 0

for i in Attribute_Names:
    Attribute_Dict[i]=counter
    counter += 1

# Read Data
# *** change data_set to Data
data_set = pd.DataFrame.from_csv("hw4-data.csv",sep = '\t')
data_set.reset_index(inplace=True)


# *** how to reduce data set
# *** feature -> Attribute, feature_value -> Mean, data_set -> Data
# reduced_data_set = data_set[data_set[feature] > feature_value]


# --------------------------- Entropy ----------------------------- #

def entropy(data_set):
    count = len(data_set)

    positive = data_set[data_set['quality'] == 1]
    positive_num = len(positive)
    negative = data_set[data_set['quality'] == 0]
    negative_num = len(negative)

    pos_percent = float(positive_num)/count
    neg_percent = float(negative_num)/count

    # log 0 is undefined
    pos_entropy = 0
    neg_entropy = 0

    if positive_num != 0:
        pos_entropy = -pos_percent * log(pos_percent, 2)
    if negative_num != 0:
        neg_entropy = -neg_percent * log(neg_percent, 2)

    total_entro = neg_entropy + pos_entropy
    return total_entro


# --------------------------- Node Structure ----------------------------- #

# Data Structure for Node in Decision Tree
class Node:
    def __init__(self, Attribute = None, Median = None, DataSet = None ):
        self.Attribute = Attribute
        self.Median = Median
        self.DataSet = DataSet
        self.Children = {}


Selected_Attributes = set()

# --------------------------- Attribute Selector ----------------------------- #

def attribute_selector(Data, Selected_Attributes):
    if len(Data) == 0:
        print 'Data not found'
        return 'No_Attr'
    min_entro = sys.maxint
    Chosen_Attr = None

    for Attribute in Attribute_Dict:
        if Attribute not in Selected_Attributes:
            Attr_Entro = 0
            Median_Val  = Data[Attribute].median()
            Data_High = Data[Data[Attribute] > Median_Val]
            Data_Low = Data[Data[Attribute] <= Median_Val]
            Attr_Entro += entropy(Data_High)
            Attr_Entro += entropy(Data_Low)
            if Attr_Entro < min_entro:
                min_entro = Attr_Entro
                Chosen_Attr = Attribute

    return Chosen_Attr

# --------------------------- quality selector ----------------------------- #

def quality_selector(Data):

    positive = len(Data[Data['quality'] == 1])
    negative = len(Data[Data['quality'] == 0])

    if positive > negative:
        return 1
    else:
        return 0


# --------------------------- classification ----------------------------- #


# --------------------------- Tree Builder ----------------------------- #

def build_tree(tree, Selected_Attributes):

    data_len = len(Node(tree).DataSet)
    positive_len  =  len(Node(tree).DataSet[tree.DataSet['quality'] == 1])
    negative_len  =  len(Node(tree).DataSet[tree.DataSet['quality'] == 1])

    if positive_len == data_len  or negative_len == data_len:
        return tree

    best_attr = attribute_selector(tree.DataSet,Selected_Attributes)
    if best_attr == 'No_Attr':
        return tree

    else:
        Node(tree).Attribute = best_attr
        Node(tree).Median = tree.DataSet[best_attr].median()
        DataLower = tree.DataSet[tree.DataSet[best_attr]<=tree.Median]
        DataHigher = tree.DataSet[tree.DataSet[best_attr]>tree.Median]
        low_node = Node(DataSet=DataLower)
        high_node = Node(DataSet=DataHigher)
        Node(tree).Children['less'] = build_tree(low_node,Selected_Attributes+[best_attr])
        Node(tree).Children['high'] = build_tree(high_node,Selected_Attributes+[best_attr])

    return tree

'''

def learn(self, training_set, col_map ):
    info_gain_dict = {}
    A = np.array(training_set)
    #print A[:,]
    result_set =  A[:,col_map['quality']]

    for i in col_map:
        temp_list = A[:,col_map[i]]
        info_gain_dict[i] = information_gain(temp_list,result_set)


# implement this function
def classify(self, test_instance):
    result = 0 # baseline: always classifies as 0
    return result

def information_gain(temp_list,result_set):
    median = median_list(temp_list)
    print median
    print result_set


def median_list(l):
    median = numpy.median(l)
    return median

def run_decision_tree():

    # Load data set

    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn( training_set , col_map )

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print "accuracy: %.4f" % accuracy       
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
'''



'''
Reading Data and Testing, Changing Code needed it
'''
accuracy=[]
dataset_length = len(data_set)
cross_valid_interval = dataset_length/10
start_test = 0
for fold in range(10):
    train_set = data_set[0:start_test].append(data_set[start_test + cross_valid_interval : dataset_length])
    test_set = data_set[start_test : start_test + cross_valid_interval]
    d_tree = build_tree(DecisionTree(data_set=train_set),[])
    test_set['classified'] = data_set.apply(get_prediction, axis=1)

    for row in data_set:
        test_set['classified'] = test_set['classified'].append(get_prediction(row))

    accuracy_value = len(test_set[test_set['classified'] == test_set['quality']]) * 1.0 /len(test_set)
    accuracy.append(accuracy_value)
    start_test += cross_valid_interval
    print accuracy

print "Final Accuracy: ", np.mean(accuracy)


'''
End
'''

