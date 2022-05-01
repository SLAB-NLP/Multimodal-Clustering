##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc


class BBoxPredictionModelWrapper:
    """ This is the base class for models for prediction of bounding boxes. """

    # Abstract methods

    @abc.abstractmethod
    def predict_bboxes(self, image_tensor=None):
        return
