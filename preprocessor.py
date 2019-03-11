import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


class Precessor:
    """"""

    def __init__(self, missing_value_method="remove", balance_method="downSampling"):
        """"""
        self.missing_value_method = missing_value_method
        self. balance_method = balance_method

    def preprocess(self, data):
        """"""

        # print(data.shape)
        data = self.__dealMissingValue(data)
        # print(data.shape)

        column2Descrtize = {"age":5, "fnlwgt":22500, "hours_per_week":5} # TODO: change names
        for col in column2Descrtize.keys():
            data = self.__discretize(data, col, column2Descrtize[col])
            # print(data[col])
            # print(data)
        # print(data.shape)

        self.__balanceDataset(data)

        return data


    def __discretize(self, data, colName, bin):
        """"""
        discretizedData = data
        min = data[colName].min()
        max = data[colName].max()
        size = max - min
        nbBucket = int(size/bin)
        newSize = nbBucket*bin + bin
        # print("{}/{} - {}({}x{}) -> {}".format(min, max, size, nbBucket, bin, newSize))
        bucketArray = np.linspace(0, newSize, nbBucket-1)
        bucketArray = [int(i) for i in bucketArray]
        discretizedData[colName] = discretizedData[colName].apply(lambda x: self.takeClosest(x, bucketArray))

        # discretizedData = pd.cut(discretizedData[colName], bins=bin, labels=np.arange(bin), right=False)
        # category = pd.cut(data[colName], bucketArray)
        # category = category.to_frame()
        # discretizedData = pd.concat([data, category.age], axis=1)
        # print(discretizedData)
        return discretizedData

    def __dealMissingValue(self, data, missing_value_method="remove"):
        """"""
        nb_row_before = data.shape[0]
        if(missing_value_method == "remove"):
            for column in data:
                data = data[data[column].notnull()]
                if(data[column].dtype == "object"):
                    data = data[data[column] != '?']

        elif(missing_value_method == "average"):
            pass
        else:
            print("Error. Missing value method unrecognized")
            exit(1)

        nb_row_remove = nb_row_before - data.shape[0]
        print("{} rows removed.".format(nb_row_remove))
        return data

    # def __dealMissingValue2(self, data, missing_value_method="remove"):
    #     """"""
    #     data = self.__convertToNumerical(data)
    #     print(data.info())
    #     if(missing_value_method == "remove"):
    #         pass
    #     return data
    #
    #
    # def __convertToNumerical(self, data):
    #     """"""
    #     data['occupation'] = data['occupation'].map({'?': 0, 'Farming-fishing': 1, 'Tech-support': 2,
    #                                                'Adm-clerical': 3, 'Handlers-cleaners': 4, 'Prof-specialty': 5,
    #                                                'Machine-op-inspct': 6, 'Exec-managerial': 7,
    #                                                'Priv-house-serv': 8, 'Craft-repair': 9, 'Sales': 10,
    #                                                'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13,
    #                                                'Protective-serv': 14}).astype(int)
    #
    #     data['income'] = data['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    #
    #
    #     data['sex'] = data['sex'].map({'Male': 0, 'Female': 1}).astype(int)
    #     data['race'] = data['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3,
    #                                     'Amer-Indian-Eskimo': 4}).astype(int)
    #
    #     data['marital_status'] = data['marital_status'].map({'Married-spouse-absent': 0, 'Widowed': 1,
    #                                                          'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,
    #                                                          'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
    #     return data




    def __balanceDataset(self, data, balance_method="downSampling", balancedClass="income"):
        """if time"""
        if(balance_method == "downSampling"):
            pass
        elif(balance_method == "overSampling"):
            pass
        else:
            print("Error. Missing value method unrecognized")
            exit(1)
        pass

    def takeClosest(self, num, collection):
       return min(collection,key=lambda x:abs(x-num))
