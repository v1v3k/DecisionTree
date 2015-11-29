import pandas as pd
import math
import sys
import numpy as np

# <codecell>

import sys
data_file_loc, = sys.argv[1:]

# This piece of code reads the csv file as a table 
data_set = pd.DataFrame.from_csv(data_file_loc, sep='\t')
data_set.reset_index(inplace=True)

# <codecell>


# Create three sets of columns, integer, float and categorical
integer_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'nr_employed']
float_columns = ['emp_var_rate', 'cons_price_idx', 'cons_conf_idf', 'euribor3m']
data_set[integer_columns] = data_set[integer_columns].astype(int)
data_set[float_columns] = data_set[float_columns].astype(float)
all_columns = set(data_set.columns)
integer_columns_set = set(integer_columns)
float_columns_set = set(float_columns)
categorical_columns_set = all_columns.difference(integer_columns_set).difference(float_columns_set)
categorical_columns_set.remove('target')

# <codecell>

# Find the distinct values of all categorical features
categorical_column_distinct_values_map = {}
for column in categorical_columns_set:
    distinct_values = data_set[column].unique()
    categorical_column_distinct_values_map[column] = distinct_values

# <codecell>

# The following piece of code calculates entropy given a set of data over a given feature and feature value
def entropy(data_set, feature, feature_value, greater=True):
    if feature in categorical_columns_set:
        reduced_data_set = data_set[data_set[feature] == feature_value]
    else:
        if greater:
            reduced_data_set = data_set[data_set[feature] > feature_value]
        else:
            reduced_data_set = data_set[data_set[feature] <= feature_value]
    if len(reduced_data_set) == 0:
        return sys.maxint
    total_count = len(reduced_data_set)
    yes_data = reduced_data_set[reduced_data_set['target'] == 'yes']
    yes_count = len(yes_data)
    no_count = total_count - yes_count
    yes_pi = yes_count * 1.0 /total_count
    no_pi = no_count * 1.0/total_count
    yes_entropy = 0.0
    no_entropy = 0.0
    if yes_count != 0:
        yes_entropy = -yes_pi * math.log(yes_pi, 2)
    if no_count != 0:
        no_entropy = - no_pi * math.log(no_pi, 2)
    entropy = yes_entropy + no_entropy
    return entropy


# <codecell>

# This peice of code selects a feature that minimizes entropy on the given data set
def select_feature(data_set, features_not_allowed_set):
    if len(data_set) == 0:
        print "ERROR: The length of dataset passed to select_feature was zero"
    # First try out the categorical features
    min_entropy_thus_far = sys.maxint


    # Now select from numerical features. We shall use the median to split
    for feature in integer_columns_set.union(float_columns_set):
        if feature not in features_not_allowed_set:
            feature_entropy = 0.0
            split_value = data_set[feature].median()
            feature_entropy += entropy(data_set, feature, split_value, greater=False)
            feature_entropy += entropy(data_set, feature, split_value, greater=True)
            if feature_entropy < min_entropy_thus_far:
                min_entropy_thus_far = feature_entropy
                selected_feature = feature 
    return selected_feature

# <codecell>

class DecisionTreeNode:
    def __init__(self, data_set= None, selected_feature= None, split_value=None):
        self.data_set = data_set
        self.selected_feature = selected_feature
        # Split value field is only for numerical features
        self.split_value = split_value
        self.children = {}
def major_label(data_set):
    yes_count = len(data_set[data_set['target'] == 'yes'])
    no_count = len(data_set) - yes_count
    if yes_count > no_count:
        return "yes"
    return "no"
def classify(root_node, data_row):
    if root_node is None:
        return None
    if len(root_node.children) == 0:
        # If this node has no children then we shall compute the label
        # by counting the majority label at this node's data
       return major_label(root_node.data_set)
    else:
        # Check which node to go to based on current node's selected feature
        selected_feature_this_node = root_node.selected_feature
        if selected_feature_this_node in categorical_columns_set:
            data_row_value_on_selected_feature = data_row[selected_feature_this_node]
            if not root_node.children.has_key(data_row_value_on_selected_feature):
                return major_label(root_node.data_set)
            next_node = root_node.children[data_row_value_on_selected_feature]
            return classify(next_node, data_row)
        else:
            if data_row[selected_feature_this_node] <= root_node.split_value:
                if not root_node.children.has_key("<="):
                    return major_label(root_node.data_set)
                next_node = root_node.children["<="]
                return classify(next_node, data_row)
            else:
                if not root_node.children.has_key("<="):
                    return major_label(root_node.data_set)
                next_node = root_node.children[">"]
                return classify(next_node, data_row)
    # Should never reach here
    print "Error: Reached impossible case in classify"
    return None

# <codecell>

# This piece of code builds a decision tree
def build_tree(tree, previously_selected_features):
    #print "Length of dataset", len(tree.data_set)
    # If the length of the data becomes too small stop
    if len(tree.data_set) < 1000:
        #print "Returning since dataset length has become less than 10"
        return tree
    # If all the labels are the same in the dataset stop
    yes_count = len(tree.data_set[tree.data_set['target'] == 'yes'])
    if len(tree.data_set) == yes_count or yes_count == 0:
        #print "Returnign since all are yeses or no's"
        return tree
    # If none of the base conditions batch we shall further split the tree
    # Select the best feature to split the tree on
    best_feature = select_feature(tree.data_set, previously_selected_features)
    if best_feature is None:
        return tree
    #print "Selected best feature ", best_feature
    tree.selected_feature = best_feature
    # Add children nodes to this tree for all non zero branches of the best feature
    # Case 1 : Best feature is categorical (categorically good I tell you)
    if best_feature in categorical_columns_set:
        distinct_values = categorical_column_distinct_values_map[best_feature]
        for value in distinct_values:
            #print "Creating subtree for value", value
            data_subset = tree.data_set[tree.data_set[best_feature] == value]
            if len(data_subset) != 0:
                child_node = DecisionTreeNode(data_set = data_subset)
                tree.children[value] = build_tree(child_node, previously_selected_features + [best_feature])
                
    # Case 2: Best feature if numerical (If it wasnt categorically good, atleast its numerically good)
    else:
        tree.split_value = tree.data_set[best_feature].median()
        #print "Split value is ", split_value
        data_subset_lesser = tree.data_set[tree.data_set[best_feature] <= tree.split_value]
        data_subset_greater = tree.data_set[tree.data_set[best_feature] > tree.split_value]
        child_node_lesser = DecisionTreeNode(data_set = data_subset_lesser) 
        child_node_greater = DecisionTreeNode(data_set = data_subset_greater)
        tree.children["<="] = build_tree(child_node_lesser, previously_selected_features + [best_feature])
        tree.children[">"] = build_tree(child_node_greater, previously_selected_features + [best_feature])
    return tree

# <codecell>

decision_tree = None
def get_prediction(df_row):
    return classify(decision_tree, df_row)

# <codecell>


