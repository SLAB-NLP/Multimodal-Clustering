##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from models_src.model_configs.model_config import ModelConfig


class TextOnlyModelConfig(ModelConfig):
    """ Contains configuration settings for the text-only baseline model.


    cluster_num: The number of clusters (for when the model is used for word clustering).

    repr_num: Number of representative words for concrete/abstract words.
    """
    def __init__(self,
                 cluster_num=41,
                 repr_num=20
                 ):
        super(TextOnlyModelConfig, self).__init__()

        self.cluster_num = cluster_num
        self.repr_num = repr_num
