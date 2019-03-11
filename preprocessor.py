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

        toDescrtize = {"a":2, "b":4} # TODO: change names
        for col in toDescrtize.keys():
            self.__discretize(data, col, toDescrtize[col])

        self.__balanceDataset(data)

        return data


    def __discretize(self, data, colName, bin):
        """"""
        pass

    def __dealMissingValue(self, data, missing_value_method="remove"):
        """"""
        if(missing_value_method == "remove"):
            data[data.workclass != "?"]
            print(data)
        elif(missing_value_method == "average"):
            pass
        else:
            print("Error. Missing value method unrecognized")
            exit(1)

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
