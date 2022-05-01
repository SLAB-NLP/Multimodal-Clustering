##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
import torch
from sklearn.cluster import KMeans
import numpy as np

# Models
from models_src.wrappers.word_clustering_model_wrapper import WordClusteringModelWrapper
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import clip


class WordEmbeddingClusteringModelWrapper(WordClusteringModelWrapper):
    """ This is the base class for models for word clustering.
        This class use models that create word embeddings, and cluster these embeddings using kmeans.
    """

    def __init__(self, device, word_list):
        self.device = device
        self.model = self.init_model()
        self.word_to_cluster = self.create_word_to_cluster(word_list)

    def create_word_to_cluster(self, word_list):
        mat_list = [self.embedding_func(x) for x in word_list]
        mat = np.concatenate(mat_list, axis=1)
        kmeans = KMeans(n_clusters=41).fit(mat.transpose())
        cluster_list = list(kmeans.labels_)
        return {word_list[i]: cluster_list[i] for i in range(len(word_list))}

    @abc.abstractmethod
    def init_model(self):
        return

    @abc.abstractmethod
    def embedding_func(self, word):
        return

    def predict_cluster_for_word(self, word):
        return self.word_to_cluster[word], 1
    
    
class W2VClusteringWrapper(WordEmbeddingClusteringModelWrapper):
    """ This class use word embedding from word2vec presented in the paper "Efficient estimation of word representations
        in vector space" by Mikolov et al.
    """
    
    def __init__(self, device, word_list):
        super(W2VClusteringWrapper, self).__init__(device, word_list)

    def init_model(self):
        return api.load("word2vec-google-news-300")

    def embedding_func(self, word):
        return self.model.word_vec(word).reshape(300, 1)


class BERTClusteringWrapper(WordEmbeddingClusteringModelWrapper):
    """ This class use word embedding from BERT presented in the paper "Bert: Pre-training of deep bidirectional
        transformers for language understand" by Devlin et al. """

    def __init__(self, device, word_list):
        super(BERTClusteringWrapper, self).__init__(device, word_list)

    def init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

    def text_preparation(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def sentence_embedding_func(self, tokens_tensor, segments_tensors):
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # Removing the first hidden state (the first state is the input state)
            hidden_states = outputs[2][1:]

        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings

    def embedding_func(self, word):
        sent = 'this is a ' + word
        tokenized_text, tokens_tensor, segments_tensors = self.text_preparation(sent)
        list_token_embeddings = self.sentence_embedding_func(tokens_tensor, segments_tensors)

        return np.array(list_token_embeddings[4]).reshape(768, 1)


class CLIPClusteringWrapper(WordEmbeddingClusteringModelWrapper):
    """ This class use word embedding from CLIP presented in the paper "Learning Transferable Visual Models From Natural
        Language Supervision" By Radford et al. """

    def __init__(self, device, word_list):
        super(CLIPClusteringWrapper, self).__init__(device, word_list)

    def init_model(self):
        return clip.load('RN50', self.device)[0]

    def embedding_func(self, word):
        return self.model.encode_text(clip.tokenize('a photo of a ' + word)).detach().numpy().transpose()
