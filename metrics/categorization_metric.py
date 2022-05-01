##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import numpy as np
from sklearn.metrics.cluster import v_measure_score
from metrics.metric import Metric


class CategorizationMetric(Metric):
    """ This metric estimates whether the textual model categorizes words according to some baseline (the category
        dataset).
        We use different measures for evaluation.
        If ignore_unknown_words==True, the model is only evaluated using words encountered in the training phase.
    """

    def __init__(self, text_model, category_dataset, ignore_unknown_words=True):
        super(CategorizationMetric, self).__init__(None, text_model)
        self.category_dataset = category_dataset
        self.ignore_unknown_words = ignore_unknown_words

        if self.ignore_unknown_words:
            self.name_prefix_str = 'ignore_unknown'
        else:
            self.name_prefix_str = 'include_unknown'

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        # This metric is not related to the test set, all the measurements are done later
        return

    def report(self):
        if self.results is None:
            self.calc_results()

        res = self.name_prefix_str + ': '
        res += 'v measure score: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_v_measure_score'] + ', '
        res += 'purity: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_purity'] + ', '
        res += 'collocation: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_collocation'] + ', '
        res += 'purity-collocation F1: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_pu_co_f1'] + ', '
        res += 'FScore: ' + \
               self.precision_str % self.results[self.name_prefix_str + '_FScore']

        return res

    def calc_results(self):
        self.results = {}

        self.compare_clusters_to_gt_classes()

        # Add unknown words prefix to all metric names
        keys_to_remove = []

        item_list = list(self.results.items())
        for key, val in item_list:
            keys_to_remove.append(key)
            self.results[self.name_prefix_str + '_' + key] = val

        for key in keys_to_remove:
            del self.results[key]

    """ Compare our induced clusters to the ground-truth classes, using different metrics. """

    def compare_clusters_to_gt_classes(self):
        gt_labels, predicted_labels = self.collect_gt_and_predictions()
        cluster_to_gt_intersection, gt_to_cluster_intersection, gt_class_count, cluster_count = \
            self.aggregate_intersection_counts(gt_labels, predicted_labels)

        # We use 3 metrics: v measure score, purity/collocation, and F1
        self.results['v_measure_score'] = v_measure_score(gt_labels, predicted_labels)
        self.results['purity'], self.results['collocation'], self.results['pu_co_f1'] = \
            self.calc_purity_collocation(cluster_to_gt_intersection, gt_to_cluster_intersection)
        self.results['FScore'] = self.calc_fscore(gt_to_cluster_intersection, gt_class_count, cluster_count)

    """ Collect the ground-truth classes and predicted clusters for all the words in the categorization dataset. """

    def collect_gt_and_predictions(self):
        gt_labels = []
        predicted_labels = []
        category_index = 0
        for category, word_list in self.category_dataset.items():
            for word in word_list:
                # Prediction
                prediction = self.text_model.predict_cluster_for_word(word)
                if prediction is not None:
                    # The word is known
                    predicted_labels.append(prediction[0])
                elif not self.ignore_unknown_words:
                    ''' The word is unknown, but we were told not to ignore unknown words, so we'll label it by a new
                    cluster. '''
                    new_cluster_ind = self.text_model.config.cluster_num
                    predicted_labels.append(new_cluster_ind)

                if (prediction is not None) or (not self.ignore_unknown_words):
                    ''' Only in case that we appended something to the predicted_labels list, we need to append to the
                    gt_labels list '''
                    gt_labels.append(category_index)

            category_index += 1

        return gt_labels, predicted_labels

    """ Calculate the size of intersection of gt classes and induced clusters. """

    @staticmethod
    def aggregate_intersection_counts(gt_labels, predicted_labels):
        cluster_to_gt_intersection = {}
        gt_to_cluster_intersection = {}
        gt_class_count = {}
        cluster_count = {}
        for i in range(len(gt_labels)):
            gt_class = gt_labels[i]
            predicted_cluster = predicted_labels[i]

            # Update counts
            if gt_class not in gt_class_count:
                gt_class_count[gt_class] = 0
            gt_class_count[gt_class] += 1
            if predicted_cluster not in cluster_count:
                cluster_count[predicted_cluster] = 0
            cluster_count[predicted_cluster] += 1

            # Update gt class to cluster mapping
            if gt_class not in gt_to_cluster_intersection:
                gt_to_cluster_intersection[gt_class] = {predicted_cluster: 0}
            if predicted_cluster not in gt_to_cluster_intersection[gt_class]:
                gt_to_cluster_intersection[gt_class][predicted_cluster] = 0
            gt_to_cluster_intersection[gt_class][predicted_cluster] += 1

            # Update cluster to gt class mapping
            if predicted_cluster not in cluster_to_gt_intersection:
                cluster_to_gt_intersection[predicted_cluster] = {gt_class: 0}
            if gt_class not in cluster_to_gt_intersection[predicted_cluster]:
                cluster_to_gt_intersection[predicted_cluster][gt_class] = 0
            cluster_to_gt_intersection[predicted_cluster][gt_class] += 1

        return cluster_to_gt_intersection, gt_to_cluster_intersection, gt_class_count, cluster_count

    """ Calculate purity, collocation and harmonic mean.
    
        Purity measures whether the members of a cluster originate from the same gt class. For a specific cluster, we
        find the gt-class with which it has the largest intersection. We sum the size of largest intersections for each
        cluster, and divide by the total number of words in the dataset.
        
        Collocation measures whether we clustered members of a gt class to a single cluster. For a specific gt class, we
        find the cluster with which it has the largest intersection. We sum the size of largest intersections for each
        class, and divide by the total number of words in the dataset.
        
        Finally, we calculate the F1-score (harmonic mean) of the two.
    """

    @staticmethod
    def calc_purity_collocation(cluster_to_gt_intersection, gt_to_cluster_intersection):
        N = sum([sum(x.values()) for x in gt_to_cluster_intersection.values()])
        if N == 0:
            return 0, 0, 0

        purity = (1 / N) * sum([max(x.values()) for x in cluster_to_gt_intersection.values()])
        collocation = (1 / N) * sum([max(x.values()) for x in gt_to_cluster_intersection.values()])
        if purity + collocation == 0:
            f1 = 0
        else:
            f1 = 2 * (purity * collocation) / (purity + collocation)

        return purity, collocation, f1

    """ Calculate the F-Score measure, defined in the SemEval 2007 task, by Agirre and Soroa.
        Given a particular class S of size Ns and a cluster H of size Nh, suppose Nsh examples in the class S belong to
        H. The F value of this class and cluster is defined to be: f(S, H) = 2P(S, H)*R(S,H) / (P(S, H) + R(S, H)) where
            - P(S, H) = Nsh/Ns is the precision value
            - R(S, H) = Nsh/Nh is the recall value
        The FScore of class S is the maximum F value attained at any cluster
        The FScore of the entire clustering solution is the sum of the FScores of classes, normalized by the relative
        size of the class.
    """

    @staticmethod
    def calc_fscore(gt_to_cluster_intersection, gt_class_count, cluster_count):
        N = sum([sum(x.values()) for x in gt_to_cluster_intersection.values()])
        if N == 0:
            return 0

        solution_Fscore = 0
        # Go over all gt classes and calculate class_Fscore for each class
        for gt_class, cluster_map in gt_to_cluster_intersection.items():
            gt_class_size = gt_class_count[gt_class]
            class_Fscore = 0
            # Go over all the clusters and calculate f value for each cluster, with the current class
            for cluster, intersection_size in cluster_map.items():
                cluster_size = cluster_count[cluster]
                precision = intersection_size / gt_class_size
                recall = intersection_size / cluster_size
                if precision + recall == 0:
                    f_value = 0
                else:
                    f_value = 2 * (precision * recall) / (precision + recall)
                if f_value > class_Fscore:
                    class_Fscore = f_value

            solution_Fscore += (gt_class_size / N) * class_Fscore

        return solution_Fscore

    @staticmethod
    def uses_external_dataset():
        return True
