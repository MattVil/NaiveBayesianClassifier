
class NaiveBayes:

    def __init__(self):
        pass

    def training(self, nbIter):
        for iter in range(nbIter):
            self.__Kfold_cross_validation()
        pass

    def __Kfold_cross_validation(self, data, trainRatio=0.5, validRatio=0.3, testRatio=0.1):
        """split dataset + compute classes probabilities"""
        xTrain, xValid, xTest = self.__split_dataset(data,
                                                     trainRatio,
                                                     validRatio,
                                                     testRatio)
        pass

    def __split_dataset(self, data, trainRatio=0.5, validRatio=0.3, testRatio=0.1):
        pass
