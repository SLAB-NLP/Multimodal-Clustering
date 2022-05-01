##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc


class ActivationMapPredictionModelWrapper:
    """ This is the base class for models for prediction of activation maps. """

    # Abstract methods

    @abc.abstractmethod
    def predict_activation_map_centers(self, image_tensor=None):
        return
