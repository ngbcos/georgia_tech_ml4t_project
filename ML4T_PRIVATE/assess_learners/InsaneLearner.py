import numpy as np
import BagLearner, LinRegLearner
class InsaneLearner(object):
    def __init__(self, **kwargs):
        self.kwargs, self.all_my_bag_learners = kwargs, []
        pass
    def author(self): return 'nlee68'
    def addEvidence(self, dataX, dataY):
        for i in range(20):
            new_bag_learner = BagLearner.BagLearner(learner=LinRegLearner.LinRegLearner, kwargs=self.kwargs, bags= 20, boost=False, verbose=False)
            new_bag_learner.addEvidence(dataX, dataY)
            self.all_my_bag_learners.append(new_bag_learner)
    def query(self, points):
        average_of_all_answers_across_bags = np.zeros(np.asarray(points).shape[0])
        for row_index in range(len(average_of_all_answers_across_bags)):
            running_average = []
            for bag in self.all_my_bag_learners:
                running_average.append(bag.query(points)[row_index])
            average_of_all_answers_across_bags[row_index] = np.mean(running_average)
        return average_of_all_answers_across_bags                       #LOOK, 20 LINES!!!!