"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import DTLearner as dt
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    leaf_sizes = np.arange(1, 250, 2)
    train_RMSE = []
    test_RMSE = []
    for leaf_size in leaf_sizes:
        learner = dt.DTLearner(leaf_size=leaf_size, verbose = True) # create a LinRegLearner
        learner.addEvidence(trainX, trainY) # train it
        print learner.author()

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "Leaf size is ", leaf_size
        print "In sample results"
        print "RMSE: ", rmse
        train_RMSE.append(rmse)
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        test_RMSE.append(rmse)
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]

        print "TERMINATE ROUND"

    #plot code
    print train_RMSE
    print test_RMSE
    train_line = plt.plot(leaf_sizes, train_RMSE, label="Train")
    test_line = plt.plot(leaf_sizes, test_RMSE, label="Test")
    # use keyword args
    plt.setp(train_line, color='r', linewidth=2.0)
    plt.setp(test_line, color = 'blue', linewidth = 2.0)

    # Now add the legend with some customizations.
    legend = plt.legend(loc='best', shadow=True)

    plt.xlabel("Leaf size")
    plt.ylabel(("RMSE"))
    plt.title("Leaf Size for DTLearner (Istanbul Data) vs. RMSE")
    plt.show()
