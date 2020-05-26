from random import seed
from random import randrange
from math import sqrt
from ..models.k_nearest_neighbors import get_neighbors
import numpy as np


class BiasedRandomForest(object):
    """
    This class implements is an biased random forest, and ensemble method that seeks to mitigate issues caused by class
    imbalances in classification problems. This this based on the method proposed by M. Bader-El-Den et al in
    "Biased Random Forest For Dealing With the Class Imbalance Problem."

    Note, the core random forest implementation is a modified version of the Random Forest implementation
    by Jason Brownlee in https://machinelearningmastery.com/implement-random-forest-scratch-python/
    """

    def __init__(self, k=10, p=0.5, min_size=1, sample_size=1.0, maximum_depth=10):
        """

        :param k: int, number of nearest neighbors
        :param p: float, critical areas ratio
        :param num_features: number of features for the split
        :param min_size: int, minimum number of samples per node
        :param sample_size: float, subsample size ratio
        :param maximum_depth: int, maximum tree depth
        """

        # random num generator seed
        seed(2)

        # set instance variables
        self._s_forest_size = 0
        self._num_features_for_split = 0
        self._p_critical_areas_ratio = p
        self._k_nearest_neighbors = k
        self._maximum_depth = maximum_depth
        self._minimum_sample_size = min_size
        self._sample_ratio = sample_size

    def fit(self, X_train, total_forest_size, max_label_set, min_label_set):

        # calculate number features
        self._num_features_for_split = BiasedRandomForest._calculate_number_features(X_train)

        shape = X_train[0].shape[0]
        # training_maj = np.empty(shape)
        # training_min = np.empty(shape)
        # training_c = np.empty(shape)

        training_maj = list()
        training_min = list()
        training_c = list()

        # split training data by min/max label sets
        for row in X_train:
            # get the target label
            curr_label = int(row[-1])
            if curr_label in max_label_set:
                training_maj.append(row)
            else:
                training_min.append(row)

        # find the potential problem areas affecting the minority instances
        for row_i in training_min:
            # update critical dataset
            training_c.append(row_i)

            # find k nearest neighbors for each minority instance in the data set
            neighbors = get_neighbors(row_i, training_maj, self._k_nearest_neighbors)

            # add unique neighbors to the critical data set
            for row_j in neighbors:
                if any((row == x).all() for x in training_c):
                    training_c.append(row_j)

        # build forest from original dataset
        self._s_forest_size = int((total_forest_size * (1 - self._p_critical_areas_ratio)))
        random_forest_1 = self._generate_forest(X_train)

        self._s_forest_size = int((total_forest_size * self._p_critical_areas_ratio))
        random_forest_2 = self._generate_forest(training_c)

        # combine the two forests to generate main forest
        #random_forest_1.append(random_forest_2)

        return random_forest_1, random_forest_2

    @staticmethod
    def _calculate_number_features(dataset):

        return int(sqrt(len(dataset[0]) - 1))

    def predict(self, node, row):
        """
        Make a prediction with a decision tree
        :param node:
        :param row:
        :return:
        """

        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def bagging_predict(self, trees, row):
        """
        Make a prediction with a list of bagged trees
        :param trees:
        :param row:
        :return:
        """

        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    @staticmethod
    def _random_subsample(sample_data, ratio):
        """
        Create a random subsample from the dataset with replacement.
        :param sample:
        :param ratio:
        :return:
        """

        subsample = list()
        num_samples = round(len(sample_data) * ratio)

        while len(subsample) < num_samples:
            index = randrange(len(sample_data))
            subsample.append(sample_data[index])

        return subsample

    @staticmethod
    def _to_terminal(group):
        """

        :param group:
        :return:
        """
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    @staticmethod
    def _gini_index(groups, classes):
        """

        :param groups:
        :param classes:
        :return:
        """

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

    @staticmethod
    def _split_feature_value(index, value, dataset):
        """

        :param value:
        :param dataset:
        :return:
        """

        left, right = list(), list()

        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)

        return left, right

    @staticmethod
    def _select_best_split_point(data, n_features):
        """
        Select the best split point for a dataset
        :param data:
        :param n_features:
        :return:
        """

        target_class_values = list(set(row[-1] for row in data))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        features = list()

        while len(features) < n_features:
            index = randrange(len(data[0]) - 1)
            if index not in features:
                features.append(index)

        for index in features:
            for row in data:
                groups = BiasedRandomForest._split_feature_value(index, row[index], data)
                gini = BiasedRandomForest._gini_index(groups, target_class_values)

                if gini < best_score:
                    best_index = index
                    best_value = row[index]
                    best_score = gini
                    best_groups = groups

        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def _split_node(self, node, max_depth, min_size, n_features, depth):
        """
         Create child splits for a node or make terminal
        :param node:
        :param max_depth:
        :param min_size:
        :param n_features:
        :param depth:
        :return:
        """

        left, right = node['groups']
        del (node['groups'])

        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = BiasedRandomForest._to_terminal(left + right)
            return

        # check for maximum depth
        if depth >= max_depth:
            node['left'] = BiasedRandomForest._to_terminal(left)
            node['right'] = BiasedRandomForest._to_terminal(right)
            return

        # process left child
        if len(left) <= min_size:
            node['left'] = BiasedRandomForest._to_terminal(left)
        else:
            node['left'] = self._select_best_split_point(left, n_features)
            self._split_node(node['left'], max_depth, min_size, n_features, depth + 1)

        # process right child
        if len(right) <= min_size:
            node['right'] = BiasedRandomForest._to_terminal(right)
        else:
            node['right'] = self._select_best_split_point(right, n_features)
            self._split_node(node['right'], max_depth, min_size, n_features, depth + 1)

    def _build_tree(self, data):
        """
        Build decision tree.
        :param data:
        :return:
        """

        initial_depth = 1

        # get root
        root = BiasedRandomForest._select_best_split_point(data,  self._num_features_for_split)

        # create child splits
        self._split_node(root, self._maximum_depth, self._minimum_sample_size, self._num_features_for_split, initial_depth)

        return root

    def _generate_forest(self, X_train):
        """
        Generates random forest.
        :param X_train: multidimensional array
        :return: list of tress (dicts)
        """

        trees = list()

        for i in range(self._s_forest_size):
            # get random subsample
            subsample = BiasedRandomForest._random_subsample(X_train, self._sample_ratio)

            # build tree from subsample
            tree = self._build_tree(subsample)
            trees.append(tree)

        return trees

