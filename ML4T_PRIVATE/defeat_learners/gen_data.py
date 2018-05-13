"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for DTs
def best4DT(seed=1489683273):
    np.random.seed(seed)
    X = np.random.randint(0, 2*math.pi, size=(500,2)) #generate random values from 0 to 2pi
    Y = np.sin(X[:,0] + X[:,1]) #nonlinear equations will mess up the linear DT
    return X, Y

#returns data that works best for LRs
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.randint(0, 2*math.pi, size=(500,2))
    Y = X[:,0] + X[:,1]  #simple addition -linear equations love it
    return X, Y

def author():
    return 'nlee68' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
