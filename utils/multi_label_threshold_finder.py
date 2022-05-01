##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.


from utils.general_utils import for_loop_with_reports, log_print
from loggable_object import LoggableObject

""" The purpose of the functions in this file is to find, given a multi-label classification problem and a model that
predicts the probability of each class for each sample, what is the best threshold, where classes with probability
lower than the threshold will be considered as negative, and classes with probability that exceeds the threshold will
be considered as positive.
The threshold is chosen to maximize the F1 score.
The input to the main function is a list of tuples: (probability, ground truth).
For example, if a sample image instantiates the "dog" class and we predicted that this class is
instantiated in this sample with probability 0.7, we'll get the tuple (0.7, True).
The output of the main function is a the best threshold. """


def find_best_threshold(prob_gt_list, indent):
    prob_gt_list.sort(key=lambda x: x[0])

    prob_threshold = choose_probability_threshold(prob_gt_list, indent)

    return prob_threshold


def choose_probability_threshold(prob_gt_list, indent):
    """ To choose the best probability threshold, we need to know how many gt and non-gt there are before and after each
    element in the list.
    So, we go over the entire list twice: once to collect how many gt and non-gt there are before each element, and
    once to collect how many there are after each element.
    One thing we need to remember is that there might be multiple probabilities with the same value in the list. So
    we need to update the count only after the last one with the same value. """
    func_name = 'choose_probability_threshold'
    prob_num = len(prob_gt_list)

    # First traverse: collect gt non gt relative to element
    log_print(func_name, indent, 'Starting forward traverse to collect gt non gt count...')
    forward_collector = GtNonGtCollector(prob_gt_list, False, indent + 1)
    gt_non_gt_count_before_element = forward_collector.collect()

    # Second traverse
    log_print(func_name, indent, 'Starting reverse traverse to collect gt non gt count...')
    reverse_collector = GtNonGtCollector(prob_gt_list, True, indent + 1)
    gt_non_gt_count_after_element = reverse_collector.collect()

    # F1 calculation for each threshold
    log_print(func_name, indent, 'Calculating F1 for each threshold...')
    best_F1 = -1
    for i in range(prob_num):
        ''' In case we choose similarity number i to be the threshold, all the gt before it will be false negative,
        all non-gt before it will be true negative, all gt after it will be true positive, and all non-gt after it
        will be false positive. '''
        tp = gt_non_gt_count_after_element[i][0]
        fp = gt_non_gt_count_after_element[i][1]
        fn = gt_non_gt_count_before_element[i][0]
        f1 = tp / (tp + 0.5 * (fp + fn))  # This is the definition of F1
        if f1 > best_F1:
            best_threshold = prob_gt_list[i][0]
            best_F1 = f1

    return best_threshold


class GtNonGtCollector(LoggableObject):
    def __init__(self, prob_gt_list, reverse, indent):
        super(GtNonGtCollector, self).__init__(indent)

        self.prob_gt_list = prob_gt_list
        self.reverse = reverse

        self.gt_non_gt_count_relative_to_element = []
        self.gt_count_so_far = 0
        self.non_gt_count_so_far = 0
        self.gt_count_from_last_different_prob = 0
        self.non_gt_count_from_last_different_prob = 0
        self.cur_prob_count = 0
        self.prev_prob = 0

    def collect(self):
        prob_num = len(self.prob_gt_list)
        if self.reverse:
            ind_start = prob_num - 1
            ind_end = -1
            step = -1
        else:
            ind_start = 0
            ind_end = prob_num
            step = 1

        for_loop_with_reports(range(ind_start, ind_end, step), prob_num, 100000,
                              self.collect_inner_loop, self.progress_report)

        # In the end, we'll have the last batch of equal probabilities, need to update for those as well
        if not self.reverse:
            self.gt_non_gt_count_relative_to_element = \
                self.gt_non_gt_count_relative_to_element + \
                [(self.gt_count_so_far, self.non_gt_count_so_far)] * self.cur_prob_count
        else:
            self.gt_count_so_far += self.gt_count_from_last_different_prob
            self.non_gt_count_so_far += self.non_gt_count_from_last_different_prob
            self.gt_non_gt_count_relative_to_element = \
                [(self.gt_count_so_far, self.non_gt_count_so_far)] * self.cur_prob_count + \
                self.gt_non_gt_count_relative_to_element

        return self.gt_non_gt_count_relative_to_element

    def collect_inner_loop(self, global_index, index_in_prob_list, print_info):
        prob, is_gt = self.prob_gt_list[index_in_prob_list]
        if global_index == 0:
            self.prev_prob = prob

        if prob != self.prev_prob:
            if not self.reverse:
                ''' In case we're going in normal direction, we don't want to include the gt and non-gt count of the
                 current probability in the list. Also, we'll add the new count at the end of the list. '''
                self.gt_non_gt_count_relative_to_element = \
                    self.gt_non_gt_count_relative_to_element + \
                    [(self.gt_count_so_far, self.non_gt_count_so_far)] * self.cur_prob_count
            self.gt_count_so_far += self.gt_count_from_last_different_prob
            self.non_gt_count_so_far += self.non_gt_count_from_last_different_prob
            if self.reverse:
                ''' In case we're going in reverse order, we want to include the gt and non-gt count of the current
                probability in the list. Also, we'll add the new count at the beginning of the list. '''
                self.gt_non_gt_count_relative_to_element = \
                    [(self.gt_count_so_far, self.non_gt_count_so_far)] * self.cur_prob_count + \
                    self.gt_non_gt_count_relative_to_element
            self.gt_count_from_last_different_prob = 0
            self.non_gt_count_from_last_different_prob = 0
            self.prev_prob = prob
            self.cur_prob_count = 0

        if is_gt:
            self.gt_count_from_last_different_prob += 1
        else:
            self.non_gt_count_from_last_different_prob += 1

        self.cur_prob_count += 1

    def progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting index ' + str(index) + ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))
