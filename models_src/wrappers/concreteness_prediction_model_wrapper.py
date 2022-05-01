##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc


class ConcretenessPredictionModelWrapper:
    """ This is the base class for models for concreteness prediction. """

    # Abstract methods

    """ Estimate concreteness for all input tokens. """

    @abc.abstractmethod
    def estimate_word_concreteness(self, sentences):
        return
