import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


class Precessor:
    """This class performe all the preprocessing steps on the dataset"""

    def __init__(self, missing_value_method="remove", balance_method="downSampling"):
        self.missing_value_method = missing_value_method
        self. balance_method = balance_method

    def preprocess(self, data):
        """Main collable methode that performe the preprocessing and return the
        cleaned data"""

        print("Missing value method: " + self.missing_value_method)
        data = self.__dealMissingValue(data, self.missing_value_method)

        column2Descrtize = {"age":5, "fnlwgt":22500, "hours_per_week":5} # TODO: change names
        for col in column2Descrtize.keys():
            data = self.__discretize(data, col, column2Descrtize[col])

        self.__balanceDataset(data)

        return data


    def __discretize(self, data, colName, bin):
        """Discretize the continous value given a bin in the Pandas array"""
        discretizedData = data
        min = data[colName].min()
        max = data[colName].max()
        size = max - min
        nbBucket = int(size/bin)
        newSize = nbBucket*bin + bin
        bucketArray = np.linspace(0, newSize, nbBucket-1)
        bucketArray = [int(i) for i in bucketArray]
        discretizedData[colName] = discretizedData[colName].apply(lambda x: self.takeClosest(x, bucketArray))

        return discretizedData

    def __dealMissingValue(self, data, missing_value_method="remove"):
        """Implement different strategy to deal with missing value :
            - remove: remove the row if there is a missing values
            - average: replace the missing value by the most frequent value"""
        nb_row_before = data.shape[0]
        if(missing_value_method == "remove"):
            for column in data:
                data = data[data[column].notnull()]
                if(data[column].dtype == "object"):
                    data = data[data[column] != '?']

        elif(missing_value_method == "average"):
            for column in data:
                if(data[column].dtype == "object"):
                    data.loc[data[column] == '?', column] = data[column].mode()[0]
        else:
            print("Error. Missing value method unrecognized")
            exit(1)

        nb_row_remove = nb_row_before - data.shape[0]
        print("{} rows removed.".format(nb_row_remove))
        return data


    def __balanceDataset(self, data, balance_method="downSampling", balancedClass="income"):
        """Implement different strategy to balance the dataset"""
        if(balance_method == "downSampling"):
            pass
        elif(balance_method == "overSampling"):
            pass
        else:
            print("Error. Missing value method unrecognized")
            exit(1)
        pass

    def takeClosest(self, num, collection):
        """return the closest value of collection from num"""
        return min(collection,key=lambda x:abs(x-num))
