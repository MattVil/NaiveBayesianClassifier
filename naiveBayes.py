from sklearn.model_selection import train_test_split

class NaiveBayes:

    def __init__(self):
        self.probabilities = []

    def training(self, data, K, trainRatio=0.5, validRatio=0.3, testRatio=0.2):
        """"""
        self.__buildProbabilities()
        self.__Kfold_cross_validation(data, K)

    def __Kfold_cross_validation(self, data, K, trainRatio=0.5, validRatio=0.3, testRatio=0.2):
        """split dataset + compute classes probabilities"""
        train, test = train_test_split(data, test_size=testRatio)

        for k in range(K):
            xTrain, xValid = self.__splitDataset(train)
            print("{}:\tTrain: {}\tVaildation: {}\tTest: {}".format(k, len(xTrain), len(xValid), len(test)))


    def __buildProbabilities(self):
        """"""
        pass

    def __splitDataset(self, data, trainRatio=0.7, validRatio=0.3):
        """"""
        return train_test_split(data, test_size=validRatio)
