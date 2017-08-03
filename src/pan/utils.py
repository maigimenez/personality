from os.path import join
from collections import Counter
from itertools import chain
from corpus import *
import xml.etree.ElementTree as ET
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def group_prediction(predictions_decoded, tweets_user):
    groups = []
    ini = 0
    for num_tweets in tweets_user:
        groups.append(np.mean(predictions_decoded[ini:ini + num_tweets]))
        ini += num_tweets

    return groups


def encode_trait(trait, training_dataset, test_dataset):
    """ Encode the traits for training and testing. """
    y_train = np.array([getattr(author.personality, trait)
                        for author in training_dataset.samples
                        for _ in author.padded])
    y_test = np.array([getattr(author.personality, trait)
                       for author in test_dataset.samples
                       for _ in author.padded])
    # Get only a label for each author
    y_test_group = np.array([getattr(author.personality, trait)
                             for author in test_dataset.samples])

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_train = encoder.transform(y_train)
    encoded_test = encoder.transform(y_test)

    oneh_train = np_utils.to_categorical(encoded_train)
    oneh_test = np_utils.to_categorical(encoded_test)

    return oneh_train, oneh_test, y_test_group, encoder


def load_dataset(dataset_filepath, truth_filepath):
    """ Given the paths where the corpus is, load it within a Corpus class

    Args:
        dataset_filepath (str): path where the PAN XMLs files are.
        truth_filepath (str): path where the ground truht is.

    Returns:
        Corpus: return all the samples within a Corpus class
    """
    dataset = Corpus()
    with open(truth_filepath) as truth_file:
        for line in truth_file.readlines():

            line_split = line.strip().split(':::')
            author_id, gender, age_range = line_split[:3]
            personality = Personality(extrovert=float(line_split[3]),
                                      stable=float(line_split[4]),
                                      agreeable=float(line_split[4]),
                                      conscientious=float(line_split[5]),
                                      open_trait=float(line_split[6]))

            # Read the tweets for the author in hand
            author_tweets_filepath = join(dataset_filepath, author_id + ".xml")
            tree = ET.parse(author_tweets_filepath)
            root = tree.getroot()
            current_author = Author(author_id, gender, age_range, personality)
            for num, child in enumerate(root):
                current_author.add(child.text)
            dataset.add(current_author)

    return dataset


def to_indexes(sentences, vocabulary):
    """ Convert a list of sentences to its corresponding indices.

    Args:
        sentences: a list where each element is a lists of words (sentences)
        vocabulary: a dictionary of words and its corresponding index

    Returns:
        a list where each element is a list of indexes (sentences)

    """
    senteces_idx = []
    for sentence in sentences:
        aux_idx = []
        for word in sentence:
            # If the word is in the vocabulary get its index otherwise use the <oov> index
            if word in vocabulary:
                word_idx = vocabulary[word]
            else:
                word_idx = vocabulary['<oov>']
            aux_idx.append(word_idx)
        senteces_idx.append(aux_idx)
    return np.array(senteces_idx)


def build_vocab(dataset):
    """ Create a lookup table and a list with the vocabulary.

    Args:
        dataset (list): a matrix with the words from the dataset

    Returns:
        vocab_sorted (list): list of the words sorted by its frequency
        lookup (dict): a dictionary with the lookup table. The keys are
            the words and the values are the indexes.
    """
    # Count how many times a word appear in the dataset
    word_counts = Counter(chain(*dataset))
    # Create a list with the most common words sorted.
    # The position will be the index of the lookup table.
    vocab_sorted = [word for word, _ in word_counts.most_common()]
    vocab_sorted.append('<oov>')
    # Create a lookup table using a dictionary. Map each index with a word
    lookup = {word: index for index, word in enumerate(vocab_sorted)}

    # TODO: Move this to the unittest
    assert len(list(lookup.keys())) == len(vocab_sorted)

    return vocab_sorted, lookup