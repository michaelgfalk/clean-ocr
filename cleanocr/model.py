"""Defines a character-level encoder-decoder model with fixed-memory attention."""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, GRU
from tensorflow.keras import Model

# pylint: disable=invalid-name;

class MyGRU(GRU):
    """GRU layer with all the necessaries."""
    def __init__(self, units):
        super(MyGRU, self).__init__(
            units=units,
            # The following parameters must be set this way
            # to use CuDNN on GPU
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=0.2,
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
            reset_after=True,
            # The following parameters are necessary for the
            # encoder-decoder architecture
            return_sequences=True,
            return_state=True,
            # Just the standard initializer
            recurrent_initializer='glorot_uniform'
        )

class FixedMemoryAttention(Layer):
    """Implements fixed-memory attention as described in Britz et al.

    Arguments:
    ==========
    K (int): the number of attention vectors to compute.

    Returns:
    ==========
    C (tensor): a K * enc_units matrix of attention vectors, where enc_units is
        the dimensionality of the hidden state of the preceding RNN layer
    """
    def __init__(self, K):
        super(FixedMemoryAttention, self).__init__()
        self.K = K

    def build(self, input_shape):
        """Create weight matrix based on dims of preceding layer."""
        # input_shape == (m, t, enc_units)
        # w_alpha are the weights for a linear transform of a GRU hidden state
        # w_alpha shape == (enc_units, K)
        # pylint: disable=attribute-defined-outside-init;
        self.w_alpha = self.add_weight("attention_weights", shape=(input_shape[-1], self.K),
                                       initializer='glorot_uniform',
                                       trainable=True)

    def call(self, inputs):
        """Propagate data through the layer."""
        # inputs = sequence of hidden states from earlier step of encoder
        # inputs shape == (m, t, enc_units)
        m, t, enc_units = inputs.shape
        K = self.K

        # Calculate the position_weights matrix (L)
        # L shape == (K, t)
        K_seq = tf.range(1, K + 1, 1, dtype=tf.float32)
        t_seq = tf.range(1, t + 1, 1, dtype=tf.float32)

        L_lhs = tf.tensordot((1 - K_seq) / K, (1 - t_seq) / t, axes=0)
        L_rhs = tf.tensordot(K_seq / K, t_seq / t, axes=0)

        L = tf.math.add(L_lhs, L_rhs)

        # Apply w_alpha to all timesteps
        # This gives each timestep a score against each of the K attention vectors.
        # Which vector should it belong to?
        # inputs, alpha -> logits ==> (m, t, enc_units), (enc_units, K) -> (m, t, K)
        alpha = tf.einsum('mtd,dk->mtk', inputs, self.w_alpha)

        # Multiply by position weights
        # This raises the score when k/K is near to t/T
        alpha = tf.einsum('mtk,kt->mtk', alpha, L)

        # Apply softmax activation in the K dimension
        # Now the distribution of each timestep among the attention vectors
        # sums to 1, such that each timestep will be weighted equally
        # when they are combined into the attention matrix C
        alpha = tf.nn.softmax(alpha, axis=-1)

        # Take linear combination hidden states weighted by score vectors
        # to produce the final attention matrix
        # hidden_states, score_vectors -> C ==> (m, t, enc_units), (m, t, K) -> (K, enc_units)
        C = tf.einsum('mtd,mtk->kd', inputs, alpha)

        return C

class Encoder(Model):
    """Implements an encoder with fixed-memory attention as described in Britz et al"""

    def __init__(self, num_chars, embedding_dim, enc_units, K):
        """Intialiser method for Encoder model"""
        super(Encoder, self).__init__()
        # Save hyperparameters
        self.num_chars = num_chars
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.K = K
        # Instantiate layers
        self.embedding = Embedding(self.num_chars, self.embedding_dim)
        self.first_gru = MyGRU(self.enc_units)
        self.second_gru = MyGRU(self.enc_units)
        self.attention = FixedMemoryAttention(self.K)

    def call(self, x, hidden):
        """Propagate data through the model"""
        # Compute hidden states of GRU
        # x shape after embedding == (m, t, embedding_dim)
        x = self.embedding(x)
        # x shape after first GRU == (m, t, enc_units)
        x = self.first_gru(x, initial_state=hidden)
        # output shape == (m, t, enc_units), state shape == (m, enc_units)
        output, state = self.second_gru(x)

        # Compute attention matrix
        # C shape == (K, enc_units)
        C = self.attention(output)

        return state, C

    def initialize_hidden_state(self, batch_size):
        """Create zero input for beginning of sequence."""
        return tf.zeros((batch_size, self.enc_units))

class Decoder(Model):
    """Implements a recurrent decoder."""

    def __init__(self, num_chars, embedding_dim, dec_units, K):
        """Initialiser method for Decoder model"""
        super(Decoder, self).__init__()
        # Save hyperparameters
        self.num_chars = num_chars
        self.dec_units = dec_units
        self.K = K
        self.embedding_dim = embedding_dim
        # Initialise layers
        self.embedding = Embedding(self.num_chars, self.embedding_dim)
        self.attention_scorer = Dense(self.K)
        self.first_gru = MyGRU(self.dec_units)
        self.second_gru = MyGRU(self.dec_units)
        self.fc = Dense(self.num_chars)

    def call(self, x, state, C):
        """Propagate data through decoder"""
        # x: output of the previous timestep [m x 1] # Expects a char idx
        # state: previous hidden state of decoder [m x dec_units]
        # C: attention matrix

        tf.assert_rank(x, 3)
        tf.assert_rank(state, 2)
        tf.assert_rank(C, 2)

        # Compute embeddings for x
        # x shape after embedding == (m, 1, embedding_dim)
        x = self.embedding(x)

        # Score attention vectors
        # beta shape == (m, k)
        beta = self.attention_scorer(state)

        # Take linear combination of attention vectors
        # beta, C -> c ==> (m, K), (K, dec_units) -> (m, dec_units)
        c = tf.matmul(beta, C)

        # Expand dims of c
        # c shape after expansion == (m, 1, dec_units)
        c = tf.expand_dims(c, 1)
        # x shape after concatenation == (m, 1, dec_units + embedding_dim)
        x = tf.concat([c, x], axis=-1)

        # Pass the concatenated vector to the GRUs
        x = self.first_gru(x)
        output, state = self.second_gru(x)

        # Remove time dimension from the output
        # output shape == (batch_size * 1, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, num_chars)
        # NB: this returns the logits
        x = self.fc(output)

        return x, state
