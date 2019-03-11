import numpy
import pandas as pd

from preprocessor import Precessor
from naiveBayes import NaiveBayes

PATH_DATA = "./data/adult.data"
DATASET_COLUMNS = ["age","workclass","fnlwgt","education","education_num",
                   "marital_status","occupation","relationship","race","sex",
                   "capital_gain","capital_loss","hours_per_week",
                   "native_country","income"]


def loadData():
    return pd.read_csv(PATH_DATA,names=DATASET_COLUMNS,sep=", ",engine='python')

def main():
    data = loadData()
    preprocessor = Precessor()
    cleanDataset = preprocessor.preprocess(data)
    print(cleanDataset.info())

    naiveBayesModel = NaiveBayes()
    naiveBayesModel.training(cleanDataset, 10)




if __name__ == '__main__':
    main()
