##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
from metrics.metric import Metric


class SentenceImageMatchingMetric(Metric):
    """ This metric chooses 2 random samples, and checks if the model knows
    to align the correct sentence to the correct image.
    This is performed by predicting the clusters for one image, and for the
    two sentences, and checking the hamming distance of the clusters vector
    predicted according to the image to that predicted according to each
    sentence. If the hamming distance is lower for the correct sentence,
    this is considered a correct prediction. """

    def __init__(self, visual_model, text_model):
        super(SentenceImageMatchingMetric, self).__init__(visual_model, text_model)
        self.correct_count = 0
        self.incorrect_count = 0
        self.overall_count = 0

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        clusters_by_image = self.visual_model.predict_cluster_indicators()
        clusters_by_text = self.text_model.predict_cluster_indicators()

        batch_size = len(text_inputs) // 2
        for pair_sample_ind in range(batch_size):
            single_sample_ind = 2 * pair_sample_ind
            sample_clusters_by_image = clusters_by_image[single_sample_ind]
            sample_clusters_by_first_caption = clusters_by_text[single_sample_ind]
            sample_clusters_by_second_caption = clusters_by_text[single_sample_ind + 1]
            first_hamming_distance = torch.sum(
                torch.abs(
                    sample_clusters_by_image - sample_clusters_by_first_caption
                )
            ).item()
            second_hamming_distance = torch.sum(
                torch.abs(
                    sample_clusters_by_image - sample_clusters_by_second_caption
                )
            ).item()

            if first_hamming_distance < second_hamming_distance:
                self.correct_count += 1
            if first_hamming_distance > second_hamming_distance:
                self.incorrect_count += 1
            self.overall_count += 1

    def calc_results(self):
        ''' We have 3 types of results: correct (correct sentence was closer), incorrect (incorrect sentence was
        closer), neither (both sentence are at the same distance).
        Accuracy is the percentage of correct results. In extended accuracy "neither" is also considered, so it's the
        percentage of results that were not incorrect. '''
        self.results = {
            'image sentence alignment accuracy': self.correct_count / self.overall_count,
            'image sentence alignment extended accuracy': 1 - (self.incorrect_count / self.overall_count)
        }

    def report(self):
        if self.results is None:
            self.calc_results()

        return 'Image sentence alignment accuracy: ' + \
               self.precision_str % self.results['image sentence alignment accuracy'] + ', ' + \
               'extended accuracy: ' + \
               self.precision_str % self.results['image sentence alignment extended accuracy']
