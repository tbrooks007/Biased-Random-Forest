# Biased Random Forest
This project provides a basic implementation of a Biased Random Forest (BRAF). BRAF is an ensemble method that seeks to mitigate issues caused by class imbalances in classification problems by "adequately representing the minority class during classification." This method was proposed by M. Bader-El-Den et al in "Biased Random Forest For Dealing With the Class Imbalance Problem."

Note, this work is for learning and POC purposes only. The implementation needs work to be production worthy.

### Dataset
Pima Indians Diabetes Database: "The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on." (Reference: Kaggle)

### Setup

1. Python 3.6+
2. virtualenv (`pip install virtualenv`)
3. virtualenvwrapper (`pip install virtualenvwrapper`)

### Install Required Libraries

- `mkvirtualenv braf` (if python 3.X+ is your default python version) OR
- Type which python3, to get the path of your python3 (i.e. /usr/local/bin/python)
- `mkvirtualenv -p [Path To Python3] braf`
- `pip install -r requirements.txt`

### Possible Improvements

There are a number of improvements that could be made to this implementation including:

- Try different feature scaling technique (i.e. mean normalization). Currently it uses the default min/max approach.
- Try out different cost functions vs Gini Index
- Try learning the optimal hyperparameters via a means like GridSearch
- Implement unit tests
- Refactor to use more object oriented approach for modeling the trees, though we may lose some performance here.
- Refactor for general run time / space complexity tuning
- Implement taking in user specified hyperparameters as cmd arguments.
- Add more model performance evaluation metrics (i.e. AUPRC and AUROC curves)
- Refactor to save trained models to disk and implement functionality that allows callers to run predictions against the BRAF model.
 
### Quick EDA Analysis

From root of the project directory, run: `python run_pima_exploratory_data_analysis.py`. This outputs an pandas 
profiler generated html file to the `/eda_output` directory.
 
### Training & Model Evaluation
From root of the project directory, run: `python train.py`.  After training is complete it logs the mean accuracy, 
test precision and test recall metrics to standard out.

#### Local Training Performance Metrics Sample (2-Folds, Forest Size: 100, K-Neighbors=100, Critial Areas Ratio: 0.5)

```
Mean Accuracy: Accuracy: 90.495%
Test Precision: 0.83
Test Recall: 0.92
```

Note, more work is needed to make training and evaluation processes more effienct and more tuning is needed as well. 


# References
M. Bader-El-Den, E. Teitei and T. Perry, "Biased Random Forest For Dealing With the Class Imbalance Problem," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 7, pp. 2163-2172, July 2019, doi: 10.1109/TNNLS.2018.2878400.
