

class KNearestNeighbors(object):
    """
    This class implements is basic k-nearest neighbor model.

    Note, the core random forest implementation is a modified version of the kNN implementation
    by Jason Brownlee in https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    """

    def __init__(self, k):
        """

        :param k: int, number of nearest neighbors
        """

        self._k_nearest_neighbors = k