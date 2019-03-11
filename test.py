import numpy
import pandas as pd

from preprocessor import Precessor
from NaiveBayes import NaiveBayes

PATH_DATA = "/home/matthieu/Dataset/adult/adult.data"
DATASET_COLUMNS = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]


def loadData():
    return pd.read_csv(PATH_DATA, names=DATASET_COLUMNS)

def main():
    data = loadData()
    print(data)
    preprocessor = Precessor()
    cleanDataset = preprocessor.preprocess(data)




if __name__ == '__main__':
    main()
