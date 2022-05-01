##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc


class WordClusteringModelWrapper:
    """ This is the base class for models for word clustering. """

    # Abstract methods

    @abc.abstractmethod
    def predict_cluster_for_word(self, word):
        return
