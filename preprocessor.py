import numpy as np
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

        self.__dealMissingValue(data)

        column2Descrtize = {"a":2, "b":4} # TODO: change names
        for col in column2Descrtize.keys():
            self.__discretize(data, col, column2Descrtize[col])

        self.__balanceDataset(data)

        return data


    def __discretize(self, data, colName, bin):
        """"""
        pass

    def __dealMissingValue(self, data, missing_value_method="remove"):
        """"""
        nb_row_before = data.shape[0]
        if(missing_value_method == "remove"):
            for column in data:
                data = data[data[column].notnull()]
                if(data[column].dtype == "object"):
                    data = data[data[column] != '?']

            # # data = data[data.age != 0]
            # data = data[data.workclass != '?']
            # # data = data[data.fnlwgt != '?']
            # # data = data[data.education != '?']
            # # data = data[data.education_num != 0]
            # # data = data[data.marital_status != '?']
            # data = data[data.occupation != '?']
            # # data = data[data.relationship != '?']
            # # data = data[data.race != '?']
            # # data = data[data.sex != '?']
            # # data = data[data.capital_gain != 0] #maybe remove 0
            # # data = data[data.capital_loss != 0] #maybe remove 0
            # # data = data[data.hours_per_week != 0]
            # data = data[data.native_country != '?']
            # # data = data[data.income != '?']
            # print(data.income.unique())

        elif(missing_value_method == "average"):
            pass
        else:
            print("Error. Missing value method unrecognized")
            exit(1)

        nb_row_remove = nb_row_before - data.shape[0]
        print("{} rows removed.".format(nb_row_remove))

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
