"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random
import copy

class QLearner(object):
    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):
        self.verbose = verbose
        self.num_actions = num_actions

        self.s = 0
        self.a = 0

        #action_list = range(self.num_actions)
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Q = [[0 for x in range(self.num_actions)] for x in xrange(self.num_states)]

        self.iterations = 0

        self.experience = []

        # for DynaQ
        if dyna > 0:
            self.R = copy.deepcopy(self.Q)  # anything to save time

            #lecture said to initialize Tc like this with a small non-zero value
            self.Tc = [[[1e-10 for a in xrange(self.num_states)] for b in xrange(self.num_actions)] for c in
                       xrange(self.num_states)]
            self.T = copy.deepcopy(self.Tc)

    # def create_state_action_library(self):
    #     return [[0 for x in range(self.num_actions)] for x in xrange(self.num_states)]

    def best_action(self, q, state):
        #find action(s) with the highest value
        indices = [i for i, x in enumerate(q[state]) if x == max(q[state])]
        action = indices[random.randrange(0, len(indices))] #choose at random if multiple actions have the same score
        return action

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = random.randint(0, self.num_actions - 1)
        self.a = action
        if self.verbose: print "s =", s, "a =", action
        return action

    #Update our Q
    def update_Q(self, s, a, s_prime, r):
        my_action = self.best_action(self.Q, s_prime)
        self.Q[s][a] = (1.0 - self.alpha) * self.Q[s][a] + self.alpha * (r + self.gamma * self.Q[s_prime][my_action])

    #This is the dyna round
    def planning(self):
        random_list = np.random.randint(len(self.experience), size=self.dyna)
        for i in range(0, self.dyna):
            temp_tuple = self.experience[random_list[i]]
            s, a, s_prime, r = temp_tuple
            self.update_Q(s, a, s_prime, r)

    #I need to include an element of randomness due to the assignment randomness requirement
    def pick_action_stochastic(self, s_prime):
        if np.random.random() < self.rar:
            return random.randrange(0, self.num_actions)
        else:
            return self.best_action(self.Q, s_prime)

    def query(self, s_prime, r):
        old_state = self.s

        #use Dyna to plan some stuff (cheap)
        if self.dyna > 0:
            self.experience.append((old_state, self.a, s_prime, r))
            self.planning()

        #update Q (expensive)
        self.update_Q(old_state, self.a, s_prime, r)

        #update State, Action
        self.s = s_prime
        self.a = self.pick_action_stochastic(s_prime)

        #update RAR
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s_prime, "a =", self.a, "Q =", self.Q[old_state][self.a]

        return self.a

    @staticmethod
    def author():
        return "nlee68"