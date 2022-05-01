##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from metrics.metric import Metric


class ClusterCounterMetric(Metric):
    """ This metric counts how many active clusters we have.
        An active cluster is a cluster with at least one word that crosses the threshold. """

    def __init__(self, text_model, token_count):
        super(ClusterCounterMetric, self).__init__(None, text_model)

        self.token_count = token_count

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    def report(self):
        if self.results is None:
            self.calc_results()

        res = 'Active cluster number: ' + str(self.results['active_cluster_num'])
        res += ', active token number: ' + str(self.results['active_token_count'])
        return res

    def calc_results(self):
        self.results = {}

        ''' Go over all the vocabulary, and for each word, check if it's associated with a cluster. If so, this cluster
        is active. '''
        active_cluster_indicators = [False] * self.text_model.config.cluster_num
        text_threshold = self.text_model.config.text_threshold
        active_token_count = 0  # In addition count the number of active tokens

        for token in self.token_count.keys():
            prediction_res = self.text_model.underlying_model.predict_cluster(token)
            if prediction_res is not None:
                predicted_cluster, prob = prediction_res
                if prob >= text_threshold:
                    active_cluster_indicators[predicted_cluster] = True
                    active_token_count += 1

        active_cluster_num = len([x for x in active_cluster_indicators if x is True])
        self.results['active_cluster_num'] = active_cluster_num
        self.results['active_token_count'] = active_token_count

    @staticmethod
    def uses_external_dataset():
        return True
