from random import seed
from random import randrange

class BiasedRandomForest(object):
    """
    This class implements is an biased random forest, and ensemble method that seeks to mitigate issues caused by class
    imbalances in classification problems. This this based on the method proposed by M. Bader-El-Den et al in
    "Biased Random Forest For Dealing With the Class Imbalance Problem."

    Note, the core random forest implementation is a modified version of the Random Forest implementation
    by Jason Brownlee in https://machinelearningmastery.com/implement-random-forest-scratch-python/
    """

    def __init__(self, num_features, min_size=1, sample_size=1.0, forest_size=100, k=10, p=0.5, maximum_depth=10):
        """

        :param forest_size: int, forest size
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
        self._s_forest_size = forest_size
        self._p_critical_areas_ratio = p
        self._num_features_for_split = num_features
        self._k_nearest_neighbors = k
        self._maximum_depth = maximum_depth
        self._minimum_sample_size = min_size
        self._sample_ratio = sample_size

    def fit(self, X_data, y_labels):
        # todo: sort out the minority class + critical version of the training data
        # todo: return combined RF as the model
        pass

