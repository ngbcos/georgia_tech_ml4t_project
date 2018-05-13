'''import DTLearner as dt
learner = dt.DTLearner(leaf_size = 1, verbose = False) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query'''

import numpy as np

class DTLearner(object):

    def __init__(self, **kwargs):
        #print "Oh hi there, I'm a deterministic learner!"
        self.leaf_size = 1 #dgaf right now
        # #kwargs['leaf_size'] #leaf_size
        pass  # move along, these aren't the drones you're looking for

    # def __init__(self, leaf_size = 1, verbose=False):
    #     #print "Oh hi there, I'm a deterministic learner!"
    #     self.leaf_size = leaf_size
    #     pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'nlee68'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # build and save the model
        self.tree = self.build_tree(dataX, dataY) #god I hope this works

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        predictions = []
        points = np.array(points)

        #print "points is of type ", type(points)
        if points.ndim > 1:
            for row in range(len(points)):
                #print "I'll predict with some row ", points[row]
                predictions.append(self.predict(self.tree, points[row]))
        else:
            #print "It's only 1 row dawg"
            point = points
            predictions.append(self.predict(self.tree, point))
        return predictions


    def split_at_value_return_children(self, idx, median_value, dataX, dataY):
        left = list()
        right = list()

        #omfg this grading function
        #merge data so the Y value travels with the X data
        # print dataX.shape
        # print dataY.shape
        dataset = np.concatenate((dataX, dataY[:,None]), axis = 1) #weird [:,None] is to add a 2nd dimension to data
        # print dataset

        for row in dataset:
            if row[idx] < median_value:
                left.append(row)
            else:
                right.append(row)
        return np.asarray(left), np.asarray(right) #testing.

    def get_split_column_and_median(self, dataframe):

        dataX = np.asarray(dataframe[:,:-1])
        dataY = np.asarray(dataframe[:,-1])

        #check df for identicality and terminate if it happens

        #This terminates the node if minimum leaf size is met
        #print "oh hi there" , len(np.unique(dataframe))
        if dataX.shape[0] == 1 or len(np.unique(dataframe)) <= self.leaf_size:
            return self.terminal_root(dataframe)

        #This terminates the node if all the Y values are the same (no need to kep calculating)
        elif dataX.shape[0] == 1 or len(np.unique(dataY)) == 1:
             return self.terminal_root(dataframe)

        correlation_coefficients = []
        # print dataframe
        index_last_column = len(dataframe[0]) - 1#first row
        # print "index is ", index_last_column
        # print type(dataframe)

        for col in range(dataX.shape[1]):
            correlation_coefficients.append(np.corrcoef(dataX[:,col], dataY)[1,0]) #gets the corrcoef and adds to array            print correlation_coefficients
        max_coeff = max(correlation_coefficients)
        idx = correlation_coefficients.index(max_coeff) #gets column number of max correlation
        median_value = np.median(dataX[:,idx])
        # print "the median value is ", median_value
        # print "split at column ", idx

        #need to do a check to see if any elements would trigger a split. In edge cases, the median doesn't cause a split
        column_vals = dataframe[:,idx]
        #print column_vals < median_value
        if len(column_vals[column_vals < median_value]) == 0 or len(column_vals[column_vals >= median_value]) == 0:
            #print "HEY!"
            #this is a termination condition. No further splitting is possible based on correlation coefficient
            return self.terminal_root(dataframe)

        children = self.split_at_value_return_children(idx, median_value, dataX, dataY)
        return {'feature': idx, 'value': median_value, 'children': children}

    def terminal_root(self, dataset):
        #assumes dataX and dataY are in 1 single dataset with dataY in the last column

        dataY = dataset[:,-1] #get last column
        return np.mean(dataY) #I think this is how the terminal value is determined

    def split(self, node):
        if isinstance(node, float):
            # print "Terminal value is ", node
            return node #returns the terminal value
        left, right = node['children']
        del(node['children'])
        #if one of the sides is empty (aka no split)
        if len(left) == 0:
            node['left'] = node['right'] = self.terminal_root(right)
        elif len(right)==0:
            node['left'] = node['right'] = self.terminal_root(left)
        if len(left) > 0:
            node['left'] = self.get_split_column_and_median(left)
            self.split(node['left'])
        if len(right) > 0:
            node['right'] = self.get_split_column_and_median(right)
            #print node['right']
            self.split(node['right'])

    def build_tree(self, dataX, dataY):
        #this should be a recursive program
        root = self.get_split_column_and_median(self.mergeXY(dataX, dataY))
        self.split(root)
        return root

    def mergeXY(self, dataX, dataY):
        # print dataX.shape
        # print dataY.shape
        return np.concatenate((dataX, dataY[:, None]), axis=1)

#print tree debugger borrowed from the web (does not impact assignment)
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * ' ', (node['feature'] + 1), node['value'])))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            pass
            # print('%s[%s]' % ((depth * ' ', node)))


    def predict(self, node, row):
        # print "Hi the row is ", row
        # print "Hi the node is ", node
        if row[node['feature']] < node['value']:
            if isinstance(node['left'], float) or isinstance(node['left'], int):
                return node['left']
            else:
                return self.predict(node['left'], row)
        else:
            if isinstance(node['right'], float) or isinstance(node['right'], int):
                return node['right']
            else:
                return self.predict(node['right'], row)


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"

