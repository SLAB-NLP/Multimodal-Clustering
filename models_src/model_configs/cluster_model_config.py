##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from models_src.model_configs.model_config import ModelConfig


class ClusterModelConfig(ModelConfig):
    """ Contains configuration settings for our clustering models.


    visual_underlying_model: The underlying visual model, used to extract the clusters vector from the raw image

    pretrained_visual_underlying_model: A flag indicating whether the visual underlying model should be pretrained on
        ImageNet

    freeze_visual_parameters: A flag indicating whether we should train the parameters of the visual network (except for
        the last layer- the last layer is trained either way). We might want to use this flag when the underlying model
        is pretrained

    visual_threshold: Given that the output of the visual encoder is a vector of size N, the input image will be
        associated with cluster i iff the i'th entry in the output vector crosses the visual_threshold

    visual_learning_rate: The learning rate for the visual encoder's training

    text_underlying_model: The underlying text model, used to extract the clusters vector from the input text

    text_threshold: Given that the output of the text encoder on a single word is a vector of size N, the word will be
        associated with cluster i iff the i'th entry in the output vector crosses the text_threshold

    cluster_num: The maximum number of clusters. This will be the size of the output of the encoders
    """
    def __init__(self,
                 visual_underlying_model='resnet50',
                 pretrained_visual_underlying_model=False,
                 freeze_visual_parameters=False,
                 visual_threshold=0.5,
                 visual_learning_rate=1e-4,
                 text_underlying_model='generative',
                 text_threshold=0.03,
                 cluster_num=100
                 ):
        super(ClusterModelConfig, self).__init__()

        self.visual_underlying_model = visual_underlying_model
        self.pretrained_visual_underlying_model = pretrained_visual_underlying_model
        self.freeze_visual_parameters = freeze_visual_parameters
        self.visual_threshold = visual_threshold
        self.visual_learning_rate = visual_learning_rate
        self.text_underlying_model = text_underlying_model
        self.text_threshold = text_threshold
        self.cluster_num = cluster_num
