import tensorflow_hub as hub
from keras import backend as K
import tensorflow as tf
from keras import Sequential, Model, Input
from keras.layers import Embedding, GRU, TimeDistributed, Activation, Dense, Bidirectional, Concatenate, RepeatVector, \
    Lambda, Dropout, concatenate, add
from keras.losses import sparse_categorical_crossentropy


def embed_gru_model(sequence_length, input_vocab_size, output_vocab_size) -> Model:
    model = Sequential()
    model.add(Embedding(input_dim=input_vocab_size, output_dim=128, input_length=sequence_length))
    model.add(GRU(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))
    model.add(TimeDistributed(Dense(output_vocab_size)))
    model.add(Activation('softmax'))

    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = "adam"

    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=['acc'])
    return model


def embed_bi_gru_model(sequence_length, input_vocab_size, output_vocab_size) -> Model:
    inputs = Input(shape=(sequence_length,))
    x = Embedding(input_dim=input_vocab_size, output_dim=128, input_length=sequence_length)(inputs)
    x = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
    outputs = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(x)
    model = Model(inputs, outputs)

    optimizer = "adam"

    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=['acc'])
    return model


def encdec_embed_gru_model(sequence_length, input_vocab_size, output_vocab_size):
    # encoding
    encoder_inputs = Input(shape=(sequence_length,))
    # Embedding
    x = Embedding(input_dim=input_vocab_size, output_dim=64, input_length=sequence_length)(encoder_inputs)
    # we use the state of the encoding GRU as the initial state of the first cell of decoding GRU chain
    # the code below performs concatination of forward and backward GRU layers
    gru_outputs, state_h = GRU(128, return_state=True, recurrent_dropout=0.2, dropout=0.2)(x)

    encoder_dense = Dropout(0.2)(Dense(128, activation='softmax', name='encoder_dense')(gru_outputs))

    encoder_outputs = RepeatVector(1)(encoder_dense)

    decoder_gru = GRU(128, return_sequences=True, return_state=True, recurrent_dropout=0.2, dropout=0.2)
    decoder_dense = Dense(output_vocab_size, activation='softmax', name='decoder_dense')

    inputs = encoder_outputs
    all_outputs = []

    # decoder chain building, where every next cell gets its state and input as the state and the output
    # from the previous cell correspondingly
    for _ in range(sequence_length):
        outputs, state_h = decoder_gru(inputs, initial_state=state_h)
        all_outputs.append(decoder_dense(outputs))
        inputs = outputs

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: concatenate(x, axis=1))(all_outputs)
    model = Model(encoder_inputs, decoder_outputs)

    # Compiling the model
    optimizer = "adam"
    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=['acc'])

    return model


def elmo_module():
    sess = tf.Session()
    K.set_session(sess)
    embed = hub.Module("elmo_module", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    return embed


def elmo_embedding(X, batch_size, sequence_length):
    elmo = elmo_module()
    return \
        elmo(
            inputs=
            {
                "tokens": tf.squeeze(tf.cast(X, tf.string)),
                "sequence_len": tf.constant(batch_size * [sequence_length])
            },
            signature="tokens",
            as_dict=True)["elmo"]


def elmo_bi_gru_model(sequence_length, batch_size, output_vocab_size):
    inputs = Input(shape=(sequence_length,), dtype=tf.string.as_numpy_dtype)
    x = Lambda(lambda X: elmo_embedding(X, batch_size, sequence_length), output_shape=(sequence_length, 1024))(inputs)
    x = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
    outputs = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(x)
    model = Model(inputs, outputs)

    optimizer = "adam"
    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=['acc'])
    return model
