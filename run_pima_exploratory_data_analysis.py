from pandas_profiling import ProfileReport
import pandas as pd
import os.path

if __name__ == "__main__":

    abs_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(abs_path, "data/diabetes.csv")
    diabetes_df = pd.read_csv(path)

    file = ProfileReport(diabetes_df)
    path = os.path.join(abs_path, 'eda_output/pima.html')
    file.to_file(output_file=path)