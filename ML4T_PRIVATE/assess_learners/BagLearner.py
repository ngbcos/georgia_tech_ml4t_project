"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
from DTLearner import DTLearner

class BagLearner(object):
    def __init__(self, learner=DTLearner, kwargs={"leaf_size":1}, bags=1, boost = False, verbose=False):
        self.learner = learner
        self.bags = bags
        self.kwargs = kwargs
        self.boost = boost
        self.verbose = verbose
        self.collection_of_models = []
        self.dataX = []
        self.dataY = []
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'nlee68'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY

        #dataframe = helpers.mergeXY(dataX, dataY)
        row_count = dataX.shape[0]
        #print row_count

        for i in range(self.bags):
            #for each bag, get some random samplings
            random_indexes = np.random.choice(row_count, row_count, replace=True)
            #print random_indexes
            bag_of_dataX = dataX[random_indexes,:]
            bag_of_dataY = dataY[random_indexes]

            #print bag_of_dataX
            #print bag_of_dataY

            #create a model with those datas
            bagged_model = self.learner(**self.kwargs)

            #add data
            bagged_model.addEvidence(bag_of_dataX, bag_of_dataY)

            #add the learned model to list of learned models (santa clause collecting models)
            self.collection_of_models.append(bagged_model)

            #debug
            #print len(self.collection_of_models)


    # def query(self, points):
    #
    #     Y_values = []
    #
    #     #iterate through our points
    #     for row in range(len(points)):
    #         #print "yo dawg ", points[row]
    #         responses = []
    #         #poll the models
    #         for model in self.collection_of_models:
    #             response = model.query(points[row])
    #             responses.append(response)
    #         Y_values.append(np.mean(responses)) #add the average of responses
    #
    #     return Y_values #I hope this works

    def query(self, points):
        #this assumes points is a list of values and not just a single value. I tested it and single values break LinRegLearner
        point_check = np.asarray(points)
        average_of_all_answers_across_bags = np.zeros(point_check.shape[0])
        responses = []
        for model in self.collection_of_models:
            response = model.query(points)
            responses.append(response)

        for query_index in range(len(average_of_all_answers_across_bags)):
            list_of_results_for_index = []
            for one_response in responses:
                list_of_results_for_index.append(one_response[query_index])
            average_of_all_answers_across_bags[query_index] = np.mean(list_of_results_for_index)

        #print "YO DUDE" , average_of_all_answers_across_bags

        # for current_array in responses:
        #     results_to_be_averaged = []
        #     for i in range(len(current_array)):
        #         results_to_be_averaged.append(current_array[i])
        #     average_of_all_answers_across_bags.append(np.mean(results_to_be_averaged))
        return average_of_all_answers_across_bags


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
