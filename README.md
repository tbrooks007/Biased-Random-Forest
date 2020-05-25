# Biased-Random-Forest
Basic implementation of Biased Random Forest (BRAF).

### Possible Improvements

There are a number of improvements that could be made to this implementation including:
- Implement feature scaling to normalize values in preprocessing phase
- Try out different cost functions vs Gini Index
- Try learning the optimal hyperparameters via a means like GridSearch
- Implement unit tests
- Refactor to use more object oriented approach for modeling the trees, though we may lose some performance here.
- Refactor for general run time / space complexity tuning
 
### Setup

1. Python 3.6+
2. virtualenv (`pip install virtualenv`)
3. virtualenvwrapper (`pip install virtualenvwrapper`)

### Install Required Libraries

- `mkvirtualenv braf` (if python 3.X+ is your default python version) OR
- Type which python3, to get the path of your python3 (i.e. /usr/local/bin/python)
- `mkvirtualenv -p [Path To Python3] braf`


# References
M. Bader-El-Den, E. Teitei and T. Perry, "Biased Random Forest For Dealing With the Class Imbalance Problem," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 7, pp. 2163-2172, July 2019, doi: 10.1109/TNNLS.2018.2878400.
