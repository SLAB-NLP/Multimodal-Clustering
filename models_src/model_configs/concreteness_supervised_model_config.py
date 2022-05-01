##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from models_src.model_configs.model_config import ModelConfig


class ConcretenessSupervisedModelConfig(ModelConfig):
    """ Contains configuration settings for the supervised concreteness model.


    use_pos_tags: Whether we should use the POS tag of a word as a feature

    use_suffix: Whether we should use common suffixes as a feature

    use_embeddings: Whether we should use pretrained word embeddings as a feature

    suffix_num: The number of suffixes we should consider as common suffixes (only used if use_suffix==True)

    fast_text_dim: The dimension of the pretrained embeddings (only used if use_embeddings==True)
    """
    def __init__(self,
                 use_pos_tags=False,
                 use_suffix=False,
                 use_embeddings=False,
                 suffix_num=200,
                 fast_text_dim=300
                 ):
        super(ConcretenessSupervisedModelConfig, self).__init__()

        self.use_pos_tags = use_pos_tags
        self.use_suffix = use_suffix
        self.use_embeddings = use_embeddings
        self.suffix_num = suffix_num
        self.fast_text_dim = fast_text_dim
