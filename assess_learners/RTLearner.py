"""
Created by Joshua Reno. 2/6/2018
"""
import numpy as np

class RTLearner(object):
    def __init__(self, verbose = False, leaf_size = 1):
        self.leaf_size = leaf_size
        self.tree = None
        pass

    def author(self):
        return 'jreno3'

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(np.concatenate((dataX, dataY[:, None]), axis=1))

    def build_tree(self, data):
        dataRow = data.shape[0]
        arr1 = np.array([[-1, np.mean(data[:,-1]), np.NAN, np.NAN]],dtype=object)
        arr2 = np.array([[-1, np.unique(data[:,-1])[0], np.NAN, np.NAN]],dtype=object)
        if dataRow <= self.leaf_size:
            return arr1
        if np.unique(data[:, -1]).shape[0] == 1:
            return arr2
        else:
            i = np.random.randint((data[:, 0:-1]).shape[1])
            data1 = data[:, i]
            split = (data[np.random.randint(dataRow), i] + data[np.random.randint(dataRow), i]) / 2
            dat = data[data1 <= split]
            if np.array_equal(dat, data):
                return arr1
            left = self.build_tree(dat)
            right = self.build_tree(data[data1 > split])
            root = np.array([[i, split, 1, left.shape[0] + 1]], dtype=object)
            return (np.concatenate((root, left, right), axis=0))

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        num = points.shape[0]
        self.list = []
        for i in range(num):
            self.help(points[i], 0)
        return self.list

    def help(self, point, index):
        var = self.tree[index, :]
        zero = var[0]
        one = var[1]
        two = var[2]
        three = var[3]
        pointZero = point[zero]
        if zero == -1:
            self.list.append(one)
            pass
        elif pointZero <= one:
            self.help(point, index + two)
        else:
            self.help(point, index + three)

if __name__=="__main__":
    print("the secret clue is 'zzyzx'")