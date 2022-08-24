import numpy
import pandas

import utils


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


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


class DecisionTree:
    def __init__(self, criterion='gini', splitter='best', max_depth=1, min_size=1):
        self._X = None
        self._y = None
        self._dataset = None
        self._root = None
        self._criterion = criterion
        self._splitter = splitter
        self._max_depth = max_depth
        self._min_size = min_size

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._dataset = self._concatenate_dataset()
        self._root = self._build_tree()
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

    def predict(self, X):
        node = self._root
        X = utils.validate_type(X)
        if isinstance(X[0], list): # is list of list
            prediction_list = list()
            for row in X:
                prediction_list.append(self._predict(node, row))
            return numpy.asarray(prediction_list)
        else:
            return numpy.asarray(self._predict(node, X))

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

    def __sklearn_is_fitted__(self):
        return True

    def _test_split(self, index, value):
        left, right = list(), list()
        for row in self._dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self._test_split(index, row[index])
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _split(self, node, depth=1):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = to_terminal(left + right)
            return
        # check for max depth
        if depth >= self._max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            return
        # process left child
        if len(left) <= self._min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = self._get_split(left)
            self._split(node['left'], depth + 1)
        # process right child
        if len(right) <= self._min_size:
            node['right'] = to_terminal(right)
        else:
            node['right'] = self._get_split(right)
            self._split(node['right'], depth + 1)

    def _build_tree(self):
        root = self._get_split(self._dataset)
        self._split(root, depth=1)
        return root

    def _concatenate_dataset(self):
        dataset = None
        if isinstance(self._X, numpy.ndarray) or isinstance(self._X, list):
            dataset = pandas.DataFrame(self._X)
        if isinstance(self._y, numpy.ndarray) or isinstance(self._y, list):
            dataset = dataset.join(pandas.DataFrame(self._y, columns=['target'])['target'])
        return dataset.values.tolist()


if __name__ == '__main__':
    X = [[2.771244718, 1.784783929, ],
         [1.728571309, 1.169761413, ],
         [3.678319846, 2.81281357, ],
         [3.961043357, 2.61995032, ],
         [2.999208922, 2.209014212, ],
         [7.497545867, 3.162953546, ],
         [9.00220326, 3.339047188, ],
         [7.444542326, 0.476683375, ],
         [10.12493903, 3.234550982, ],
         [6.642287351, 3.319983761, ]]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.20, random_state=42)
    model = DecisionTree(criterion='gini', splitter='best', max_depth=3, min_size=1)
    model.fit(X_train,y_train)
    print('Score %s' % model.score(X_train, y_train))

    # DATASET
    df = pandas.read_csv('../data_banknote_authentication.csv', header=None)
    X = df.iloc[:, :4]
    y = df.iloc[:, -1]
    y = utils.label_encoder(y)
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.33, random_state=42)

    model = DecisionTree(criterion='gini', splitter='best', max_depth=5, min_size=10)
    model.fit(X_train, y_train)
    print('Score %s' % model.score(X_test, y_test))