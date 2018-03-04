"""
Created by Joshua Reno. 2/6/2018
"""
import numpy as np

class BagLearner(object):
    def __init__(self, bags, kwargs, learner, verbose = False, boost = False):
        self.bags = bags
        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'jreno3'

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        for i in range(self.bags):
            self.learners[i].addEvidence(dataX, dataY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        list = []
        for i in range(self.bags):
            list.append(self.learners[i].query(points))
        return np.mean(list, 0)

if __name__=="__main__":
    print ("the secret clue is 'zzyzx'")