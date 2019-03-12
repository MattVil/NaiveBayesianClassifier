import numpy as np
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

    print("\n############### PROPROCESS DATA ##################")
    preprocessor = Precessor(missing_value_method="remove")
    cleanDataset = preprocessor.preprocess(data)

    print("\n###### DATA INFORMATION AFTER PREPROCESSING ######")
    print(cleanDataset.info())

    naiveBayesModel = NaiveBayes()
    naiveBayesModel.Kfold_cross_validation(cleanDataset, 10)

    naiveBayesModel.plotConfusionMatrix()




if __name__ == '__main__':
    main()
