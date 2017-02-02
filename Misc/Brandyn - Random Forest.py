# Random Forest 
from random import seed
from random import randrange
from math import sqrt
import numpy as np




dataset = np.loadtxt(open('iris.csv','rb'),delimiter=',')
dataset = dataset.tolist()
split = int(len(dataset)/2)
train = dataset[:split]
test = dataset[split:]
actual = [row[-1] for row in test]



def accuracy_check(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


# Split an array based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = [], []
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini
 
# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	#print("Class values:", class_values)
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = []
	#print(features)
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1) #Randomly grab paramters
		if index not in features:
			features.append(index) #If not in already then append
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			#print("GROUPS:",groups)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	#print("Outcomes:", outcomes)
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(dataset, n_features)
	#print("Root:", root)
	split(root, max_depth, min_size, n_features, 1)
	#print("Split thing:", split)
	return root
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Create a random subsample from the dataset with replacement
def subsample(dataset, sample_size):
	sample = []
	n_sample = round(len(dataset) * sample_size)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample
 
# Make a prediction of each tree in the forest.
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count) #Find the most predicted outcome
 
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		#print("TREE:",tree)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)


def main():
    seed(0)
    max_depth = 10
    min_size = 2
    n_features = int(sqrt(len(dataset[0])-1))
    sample_size = 1.0
    scores = []
    for n_trees in [1,2,5,10,25,50,500]:
        predicted = random_forest(train,test,max_depth,min_size,sample_size,n_trees,n_features)
        accuracy = accuracy_check(actual,predicted)
        scores.append((n_trees,accuracy))
    for score in scores:
        print("Trees: ", str(score[0]) + " " + "Accuracy: ", str(score[1]))
    



