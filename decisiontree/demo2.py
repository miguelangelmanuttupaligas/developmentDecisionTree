import numpy
import pandas
import utils


# Example of building a tree
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        print(f"Característica: {get_feature(index)}")
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print(f"Valor Gini en Row: {row} es {gini}")
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'izq': b_groups[0], 'der': b_groups[1], 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
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
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def concatenate_dataset(X, y):
    dataset = None
    if isinstance(X, numpy.ndarray) or isinstance(X, list):
        dataset = pandas.DataFrame(X)
    if isinstance(y, numpy.ndarray) or isinstance(y, list):
        dataset = dataset.join(pandas.DataFrame(y, columns=['target'])['target'])
    return dataset.values.tolist()


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


def get_feature(index):
    features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    return features[index]


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[%s < %.4f] División: izq: %i, der: %i' % (depth * '   ', (get_feature(node['index'])), node['value'], len(node['izq']), len(node['der'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % (depth * '   ', node))


class DecisionTreeStatic:
    def __init__(self, max_depth=1, min_size=1):
        self._X = None
        self._y = None
        self._dataset = None
        self._root = None
        self._max_depth = max_depth
        self._min_size = min_size

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._dataset = concatenate_dataset(self._X, self._y)
        self._root = build_tree(self._dataset, self._max_depth, self._min_size)
        return self._root

    def score(self, X, y):
        X = utils.validate_type(X)
        y = utils.validate_type(y)
        correct = 0
        index = 0
        for row in X:
            prediction = self.predict(row)
            if y[index] == prediction:
                correct += 1
            index += 1
        return correct / float(len(X)) * 100.00

    def __sklearn_is_fitted__(self):
        return True

    def _predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

    def predict(self, X):
        node = self._root
        X = utils.validate_type(X)
        if isinstance(X[0], list):  # is list of list
            prediction_list = list()
            for row in X:
                prediction_list.append(self._predict(node, row))
            return numpy.asarray(prediction_list)
        else:
            return numpy.asarray(self._predict(node, X))
