##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

class ModelConfig:
    """ Contains configuration settings for our models. """
    def __init__(self):
        return

    def __str__(self):
        return 'Configuration: ' + str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__
