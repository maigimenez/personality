import numpy as np

from gensim.models import word2vec

from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, \
    Flatten, Input, Merge, Convolution1D, MaxPooling1D, Concatenate


def cnn_model(vocab_sorted, sequence_length, embedding_dim, num_filters, filters_h, model_variation,
              dropout_prob, hidden_dims, output_neurons, verbose=False, embedding_weights=None):

    # Input
    if model_variation == "CNN-static":
        input_shape = (sequence_length, embedding_dim)
        model_input = Input(shape=input_shape)
        embedding_input = model_input
    else:
        input_shape = (sequence_length,)
        model_input = Input(shape=input_shape)
        embedding_input = Embedding(len(vocab_sorted), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights)(model_input)
        print("Embedding dim:", len(vocab_sorted), embedding_dim)

    dropout_input = Dropout(dropout_prob[0])(embedding_input)

    # CNNs
    convs = []
    for fh in filters_h:
        # TODO Change the stride
        conv = Convolution1D(filters=num_filters,
                             kernel_size=fh,
                             padding='valid',
                             activation='relu',
                             strides=1)(dropout_input)
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    concat_layer = Concatenate(axis=-1)(convs)
    dropout_cnn = Dropout(dropout_prob[1])(concat_layer)

    h1 = Dense(hidden_dims[0], activation="relu")(dropout_cnn)
    h2 = Dense(hidden_dims[1], activation="relu")(h1)
    model_output = Dense(output_neurons, activation="sigmoid")(h2)

    model = Model(inputs=model_input, outputs=model_output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mean_squared_error])
    if verbose:
        print(model.summary())
    return model