from os.path import join
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

from utils import *
from cnn import cnn_model


def load_data():
    PAN_DATASET = "/Users/maite/Dev/corpora/PAN/"
    DATASET_FILEPATH = join(PAN_DATASET, "pan15-author-profiling-training-dataset-english-2015-04-23")
    TRUTH_FILEPATH = join(DATASET_FILEPATH, "truth.txt")
    training_dataset = load_dataset(DATASET_FILEPATH, TRUTH_FILEPATH)

    TEST_FILEPATH = join(PAN_DATASET, "pan-ap2015-test")
    DATASET_FILEPATH = join(TEST_FILEPATH, "en")
    TRUTH_FILEPATH = join(TEST_FILEPATH, "en.txt")
    test_dataset = load_dataset(DATASET_FILEPATH, TRUTH_FILEPATH)

    training_str = 'The mininum numbers of tweets for an author is {} in training and {} in test'
    print(training_str.format(training_dataset.min(), test_dataset.min()))
    testing_str = 'The maximum numbers of tweets for an author is {} in training and {} in test'
    print(testing_str.format(training_dataset.max(), test_dataset.max()))

    return training_dataset, test_dataset


def read_tweets(training_dataset, test_dataset):
    # TODO Check this, I think there is some new way to do it.
    # Pad the training and test dataset, to fix the input size of the data
    max_sequence = training_dataset.pad()
    _ = test_dataset.pad(max_sequence)

    # Get tweets for each author as an independent sample
    x_train = [tweet for author in training_dataset.samples for tweet in author.padded]
    x_test = [tweet for author in test_dataset.samples for tweet in author.padded]
    tweets_user = np.array([len(author.padded) for author in test_dataset.samples])

    return x_train, x_test, tweets_user


def prepare_data(x_train, x_test, lookup):

    # TODO complete this docstring
    """ Convert words into its indices

    Args:
        x_train:
        x_test:
        lookup:

    Returns:

    """
    x_train_idx = to_indexes(x_train, lookup)
    x_test_idx = to_indexes(x_test, lookup)
    print("Number of training samples: {}".format(x_train_idx.shape[0]))
    print("Number of test samples: {}".format(x_test_idx.shape[0]))
    print("Number of words per sample: {}".format(x_train_idx.shape[1]))
    assert x_train_idx.shape[1] == x_test_idx.shape[1]

    return x_train_idx, x_test_idx


def classification(seq_len, vocab_sorted, x_train_idx, x_test_idx,
                   oneh_train, oneh_test, trait, encoder, verbose):

    model_variation = 'CNN-rand'  # CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    sequence_length = seq_len
    embedding_dim = 50
    num_filters = 100
    dropout_prob = (0.25, 0.5)
    hidden_dims = (1024, 128)
    filters_h = (1, 3, 4, 5, 7)

    batch_size = 100
    num_epochs = 10
    output_classes = 9 if trait!='open' else 8

    model = cnn_model(vocab_sorted, sequence_length, embedding_dim,
                      num_filters, filters_h, model_variation,
                      dropout_prob, hidden_dims, output_classes, verbose)

    # model.fit(x_train_idx[:13000], oneh_train[:13000],
    #           batch_size=batch_size, epochs=num_epochs,
    #           validation_data=(x_train_idx[13000:], oneh_train[13000:]))

    model.fit(x_train_idx, oneh_train,
              batch_size=batch_size, epochs=num_epochs)

    loss, rmse = model.evaluate(x_test_idx, oneh_test, verbose=0)
    rmse_str = 'RMSE for trait {}: {:.4f} RMSE/per trait'
    print(rmse_str.format(trait, sqrt(rmse)))

    predictions = model.predict(x_test_idx)
    predictions_decoded = encoder.inverse_transform([np.argmax(p)
                                                     for p in predictions])
    return predictions_decoded


def rmse_grouped(trait, predictions, tweets_user, test_grouped):
    grouped_predictions = group_prediction(predictions, tweets_user)
    rmse_person = np.sqrt(mean_squared_error(test_grouped, grouped_predictions))
    print("The RMSE for the trait {} is {:.4f}".format(trait, rmse_person))
    return rmse_person


if __name__ == "__main__":
    np.random.seed(2)

    # Load the datasets to train and test the models
    training_dataset, test_dataset = load_data()
    x_train, x_test, tweets_user = read_tweets(training_dataset, test_dataset)
    # Get the length of the sequence. Since it's padded, each sentence has the same length.
    seq_len = len(x_train[0])
    # Learn a vocabulary and create the lookup table
    vocab, lookup = build_vocab(x_train)
    print("Vocabulary Size: {:d}".format(len(vocab)))
    x_train_idx, x_test_idx, = prepare_data(x_train, x_test, lookup)

    traits = ['extroverted', 'stable', 'agreeable', 'conscientious', 'open']
    rmses = []
    verbose = True
    for trait in traits:
        oh_train, oh_test, test_grouped, encoder = encode_trait(trait,
                                                                training_dataset,
                                                                test_dataset)

        predictions = classification(seq_len, vocab, x_train_idx, x_test_idx,
                                     oh_train, oh_test, trait, encoder, verbose)
        rmses.append(rmse_grouped(trait, predictions, tweets_user, test_grouped))
        verbose = False

    print("The RMSE for the individual traits are:")
    for tr_pos, trait in enumerate(traits):
        print("    - Trait {} is {:.4f}".format(trait.upper(), rmses[tr_pos]))
    print("The mean RMSE of the system is {:.4f}".format(np.mean(np.array(rmses))))



