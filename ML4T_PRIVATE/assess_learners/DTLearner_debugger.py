from DTLearner import DTLearner
from RTLearner import RTLearner
from BagLearner import BagLearner
import pandas
import os
from LinRegLearner import LinRegLearner
import numpy as np

#load some data

script_dir = os.path.dirname(__file__)
rel_path_train = "Data/simple.csv"
abs_file_path = os.path.join(script_dir, rel_path_train)
url = abs_file_path

print abs_file_path
print url


names = ['X0', 'X1', 'Y']

index_last_column = len(names) - 1

def get_xy_from_csv(url, names = None):
    if names is None:
        dataframe = pandas.read_csv(url) #names=names) #kaggle data already has the columns so...
    else:
        dataframe = pandas.read_csv(url, names = names)
    array = dataframe.values
    X = array[:,0:index_last_column]
    Y = array[:,index_last_column]
    return X, Y, dataframe

dataX, dataY, dataframe = get_xy_from_csv(url, names = names)
#

kwargs = {'leaf_size':1, 'verbose':False}
shiny_DTLearner = DTLearner(**kwargs)
shiny_DTLearner.addEvidence(dataX, dataY)

shiny_RTLearner = RTLearner(**kwargs)
shiny_RTLearner.addEvidence(dataX, dataY)

shiny_DTLearner.print_tree(shiny_DTLearner.tree)
shiny_RTLearner.print_tree(shiny_RTLearner.tree)

row = [[3, 3], [0, 5], [1, 3]]

print "deterministically " , shiny_DTLearner.query(row)
print "randomly ", shiny_RTLearner.query(row)

learner = BagLearner(learner = DTLearner, kwargs = {"leaf_size":1, "verbose": False}, bags = 20, boost = False, verbose = False)
learner.addEvidence(dataX, dataY)
print learner.query(row)

learner = BagLearner(learner = RTLearner, kwargs = {"leaf_size":1, "verbose": False}, bags = 20, boost = False, verbose = False)
learner.addEvidence(dataX, dataY)
print learner.query(row)

myLinReg = LinRegLearner()
myLinReg.addEvidence(dataX, dataY)
print myLinReg.query([[3,3], [4,4]])

mybag = BagLearner(learner=LinRegLearner, kwargs={}, bags= 20, boost=False, verbose=False)
mybag.addEvidence(dataX, dataY)
print "holy crap does it work?" , mybag.query(row)

import InsaneLearner as it
learner = it.InsaneLearner(verbose = False) # constructor
learner.addEvidence(dataX, dataY) # training step
Y = learner.query(row) # query
print "This is my answer" , Y
