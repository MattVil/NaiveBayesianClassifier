import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NaiveBayes:
    """Use the Naive Bayes algorithm and K-Fold cross-Validation on a dataset"""

    def __init__(self):
        self.probabilities = {}
        self.confusionMatrix = None
        self.removedCol = ['income']

    def Kfold_cross_validation(self, data, K, trainRatio=0.5, validRatio=0.3, testRatio=0.2):
        """split dataset and repeate K times the cross validation"""
        train, test = train_test_split(data, test_size=testRatio)

        print("\n############# K-FOLD CROSS-VALIDATION #############")
        meanConfusionMatrix = np.zeros([2, 2], dtype = int)
        for k in range(K):
            xTrain, xValid = self.__splitDataset(train, validRatio=0.375)
            print("{}:\tTrain: {}\tVaildation: {}\tTest: {}".format(k, len(xTrain), len(xValid), len(test)))

            self.__buildProbabilities(xTrain)
            confusion = self.__test(xValid)
            meanConfusionMatrix += confusion
            print(confusion)

        print("\n################# MEAN VALIDATION #################")
        meanConfusionMatrix = meanConfusionMatrix / K
        truePos = meanConfusionMatrix[0][0]
        falsePos = meanConfusionMatrix[1][0]
        trueNeg = meanConfusionMatrix[1][1]
        falseNeg = meanConfusionMatrix[0][1]
        precision = truePos / (truePos + falsePos)
        recall = truePos / (truePos + falseNeg)
        f1 = (2 * precision * recall)/(precision + recall)
        print("Precision: {}\tRecall: {}\tF1: {}".format(precision, recall, f1))
        wellClassified = truePos + trueNeg
        badClassified = falseNeg + falsePos
        totalClassified = wellClassified + badClassified
        print("Good classification: {:.2f}%\tBad classification: {:.2f}%".format((wellClassified/totalClassified)*100, 
                                                                               (badClassified/totalClassified)*100))
        self.confusionMatrix = meanConfusionMatrix.astype(int)

        print("\n######################  TEST  #####################")
        confusionMatrix = np.zeros([2, 2], dtype = int)
        confusionMatrix = self.__test(test)
        truePos = confusionMatrix[0][0]
        falsePos = confusionMatrix[1][0]
        trueNeg = confusionMatrix[1][1]
        falseNeg = confusionMatrix[0][1]
        precision = truePos / (truePos + falsePos)
        recall = truePos / (truePos + falseNeg)
        f1 = (2 * precision * recall)/(precision + recall)
        print("Precision: {}\tRecall: {}\tF1: {}".format(precision, recall, f1))
        wellClassified = truePos + trueNeg
        badClassified = falseNeg + falsePos
        totalClassified = wellClassified + badClassified
        print("Good classification: {:.2f}%\tBad classification: {:.2f}%".format((wellClassified/totalClassified)*100, 
                                                                               (badClassified/totalClassified)*100))



    def __test(self, data):
        """Use the table of probabilities made on the training set to look the
        accuracy on the test set (data)"""
        confusion = np.zeros([2, 2], dtype = int)
        for index, row in data.iterrows():
            probOver, probBelow = 1, 1
            for col in data:
                if(col not in self.removedCol):
                    try:
                        probOver *= self.probabilities[col][row[col]][0]
                    except KeyError:
                        probOver *= 1/5000
                    try:
                        probBelow *= self.probabilities[col][row[col]][1]
                    except KeyError:
                        probBelow *= 1/13000
            if(probOver > probBelow):
                if(row['income'] == ">50K"):
                    confusion[0][0] += 1
                else:
                    confusion[0][1] += 1
            else:
                if(row['income'] == "<=50K"):
                    confusion[1][1] += 1
                else:
                    confusion[1][0] += 1

        return confusion



    def __buildProbabilities(self, data):
        """Build the table of probabilities according to the training data"""
        self.probabilities = {}
        nbData = len(data)
        overData = data[data.income == '>50K']
        belowData = data[data.income == '<=50K']
        nbOver = len(overData)
        nbBelow = len(belowData)
        for column in data:
            items = data[column].unique()
            colDic = {}
            for item in items:
                try:
                    nbItemOver = overData[column].value_counts()[item]
                except KeyError:
                    nbItemOver = 1
                try:
                    nbItemBelow = belowData[column].value_counts()[item]
                except KeyError:
                    nbItemBelow = 1

                colDic[item] = ((nbItemOver/nbOver), (nbItemBelow/nbBelow))
            self.probabilities[column] = colDic

    def __splitDataset(self, data, trainRatio=0.7, validRatio=0.3):
        """Split a dataset into 2 sub-set"""
        return train_test_split(data, test_size=validRatio)

    def plotConfusionMatrix(self):
        df_cm = pd.DataFrame(self.confusionMatrix, [">50K", "<=50K"], [">50K", "<=50K"])
        plt.figure(figsize = (10,7))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
        plt.show()

    def setRomovedCol(self, col2remove):
        self.removedCol += col2remove
