##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc


class Metric:
    """ This class represents a metric for evaluating the model.
    It is given the model and inputs, generates a prediction for the specific metric, compares to
    the ground-truth and reports the evaluation of the specific metric. """

    def __init__(self, visual_model, text_model):
        self.visual_model = visual_model
        self.text_model = text_model
        self.results = None
        self.precision_str = "%.4f"

    """ Predicts the output of the model for a specific input, compares
    to ground-truth, and documents the current evaluation. """

    @abc.abstractmethod
    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    """ Reports some aggregation of evaluation of all inputs. """

    @abc.abstractmethod
    def report(self):
        return

    """ Returns a mapping from metric name to metric value. """

    @abc.abstractmethod
    def calc_results(self):
        return

    """ A flag to indicate whether this metric is only related to images. """

    @staticmethod
    def is_image_only():
        return False

    """ A flag to indicate whether this metric uses an external dataset. """

    @staticmethod
    def uses_external_dataset():
        return False
