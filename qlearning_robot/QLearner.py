"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions

        #############
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.zeros((self.num_states, self.num_actions))
        self.s = 0
        self.a = 0
        if self.dyna != 0:
            self.Tc = np.ndarray(shape=(num_states, num_actions, num_states))
            self.Tc.fill(0.00001)
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R = np.ndarray(shape=(num_states, num_actions))
            self.R.fill(-1.0)
        #############

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)

        #############
        if rand.random() > self.rar:
            action = np.argmax(self.q[s,])
        #############

        if self.verbose: print("s =", s,"a =",action)
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action = rand.randint(0, self.num_actions-1)

        #############
        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime,]))
        if rand.random <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[s_prime,])
        # self.rar = self.rar * self.radr
        self.rar *= self.radr

        if self.dyna:
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + (self.alpha * r)
            for i in range(0, self.dyna):
                a = np.random.randint(low=0, high=self.num_actions)
                s = np.random.randint(low=0, high=self.num_states)
                self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (self.R[s, a] + self.gamma * np.max(self.q[np.random.multinomial(1, self.T[s, a,]).argmax(),]))
        self.s = s_prime
        self.a = action
        #############

        if self.verbose: print( "s =", s_prime,"a =",action,"r =",r)
        return action

    def author(self):
        return 'jreno3'

if __name__=="__main__":
    print ("Remember Q from Star Trek? Well, this isn't him")
