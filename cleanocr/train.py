"""Defines training loop for model."""

# pylint: disable=invalid-name;

import os
import time
import pickle as p

from importlib_resources import path
import tensorflow as tf

from cleanocr.preprocess import load_dataset
from cleanocr.model import Encoder, Decoder
from cleanocr.utils import format_time, HParams

class Trainer():
    """Class for orgainising training loop."""
    def __init__(
            self,
            hparams,
            load_saved_model=True,
            load_saved_data=True
        ):

        # Type checks
        assert isinstance(hparams, HParams)
        assert isinstance(load_saved_model, bool)
        assert isinstance(load_saved_data, bool)

        # Hyperparameters
        self.hparams = hparams

        # Data-derived hyperparameters
        self.tkzr = None
        self.seqs = None
        self.train_data = None
        self.val_data = None
        self.bucket_info = None
        self.encoder = None
        self.decoder = None
        self.checkpoint = None

        # Loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

        # Build model
        if load_saved_data:
            self._load_saved_data()
        else:
            x_path = os.path.join(self.hparams.data_dir, self.hparams.x_path)
            y_path = os.path.join(self.hparams.data_dir, self.hparams.y_path)
            self._load_raw_data(x_path, y_path)

        self._define_model()

        if load_saved_model:
            self._load_saved_variables()
        else:
            self._create_checkpoint_dir()

    # Functions for training loop
    def loss_function(self, real, pred):
        """Applies categorical crossentropy, ignoring padding.

        Arguments:
        ==========
        `real` (tensor): the correct values
        `pred` (tensor): the model's predictions"""
        # Mask: ignore the model's predictions where the ground truth is padding
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        # Calculate the loss
        loss_ = self.loss_object(real, pred)

        # Make mask compatible with the loss output
        mask = tf.cast(mask, dtype=loss_.dtype)

        # Multiply the losses by the mask (i.e. zero out all losses where there's just padding)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @staticmethod
    def sample(pred, real, teacher_force_prob):
        """Samples from previous timestep with given probability.

        Arguments:
        ==========
        pred (tensor): [m x n] tensor of output from previous timestep
        real (tensor): [m x n] tensor of correct outputs
        teacher_force_prob (tensor): float, probability of using target letter as
            next input

        Returns:
        ==========
        dec_input (tensor): [m x 1] tensor of mixed target tokens and sampled ones
        """

        # NB: tf.random.categorical works with log probabilities

        batch_size = pred.shape[0]

        # Sample next input from predictions
        samples = tf.random.categorical(pred, num_samples=1, dtype=tf.int32)

        # Create mask using teacher_force_prob
        # Copy the probabilities m times
        mask = tf.tile([tf.math.log(1-teacher_force_prob), tf.math.log(teacher_force_prob)],
                       [batch_size])
        # Reshape into an m x 2 tensor
        mask = tf.reshape(mask, [batch_size, 2])
        # Sample to create mask
        mask = tf.random.categorical(mask, num_samples=1)
        # Recast into a boolean tensor
        mask = tf.cast(mask, dtype=tf.bool)

        # Update samples with true value where mask says to do so
        dec_input = tf.where(mask, real, samples)

        return dec_input

    @tf.function
    def train_step(self, inp, targ, teacher_force_prob, enc_hidden, out_len):
        """Performs forward- and back-propagation on a single training batch."""
        loss = 0

        with tf.GradientTape() as tape:

            # Encode input sequence: retrieve last hidden state and attention matrix
            enc_hidden, C = self.encoder(inp, enc_hidden)
            print(f'enc_hidden.shape: {enc_hidden.shape}')
            print(f'C.shape: {C.shape}')

            # Set initial state of decoder to same as encoder
            dec_hidden = enc_hidden
            print(f'dec_hidden.shape: {dec_hidden.shape}')

            # Set initial input of decoder to initial input of input sequence
            dec_input = tf.expand_dims(inp[:, 0], 1)
            print(f'dec_input.shape: {dec_input.shape}')

            for t in range(1, out_len):
                # passing enc_output to the decoder
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden, C)

                loss += self.loss_function(targ[:, t], predictions)
                self.update_accuracy(targ[:, t], predictions, self.train_acc)

                # Teacher forcing/scheduled sampling
                dec_input = self.sample(predictions, targ[:, t], teacher_force_prob)
                print(f'Shape of dec_input after sampling: {dec_input.shape}')

        self.train_loss.update_state(loss)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        # Clip gradients
        clipped_gradients = [tf.clip_by_norm(grad, self.hparams.norm_lim) for grad in gradients]

        self.hparams.optimizer.apply_gradients(zip(clipped_gradients, variables))

        return self.train_loss.result(), self.train_acc.result()

    @tf.function
    def val_step(self, inp, targ, teacher_force_prob, enc_hidden, out_len):
        """Perform forward-propagation on a single validation batch."""
        loss = 0

        # Begin feeding data to network
        enc_hidden, C = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(inp[:, 0], 1)

        # Cycle through the rest of the time steps
        for t in range(1, out_len):
            # Pass enc_output to the decoder
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden, C)

            # Calculate loss and acc
            loss += self.loss_function(targ[:, t], predictions)
            self.update_accuracy(targ[:, t], predictions, self.val_acc)

            # Use teacher forcing or sampling as appropriate...
            dec_input = self.sample(predictions, targ[: t], teacher_force_prob)

        # Calculate val_loss
        self.val_loss.update_state(loss)

        return self.val_loss.result(), self.val_acc.result()

    @staticmethod
    def update_accuracy(real, pred, acc_object):
        """Updates accuracy metric, ignoring padding variable."""

        # Find padding
        mask = tf.math.greater(real, [0]) # 'True' for all non-zero values
        mask = tf.cast(mask, dtype=tf.dtypes.int8) # 'True'  = 1, 'False' = 0

        # Compute accuracy
        acc_object.update_state(real, pred, sample_weight=mask)

    def train(self, epochs=None): # pylint: disable=too-many-locals;
        """Run training loop"""

        # Teacher forcing hparams
        _force_decay = self.hparams.teacher_force_decay
        _force_prob = self.hparams.teacher_force_prob

        # Either use passed number of epochs, or look up hparams
        if epochs is not None:
            epochs_ = epochs
        else:
            epochs_ = self.hparams.epochs

        # Loop over epochs
        for epoch in range(epochs_):
            print(f'Starting Epoch {epoch + 1}\n')

            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.val_loss.reset_states()
            self.val_acc.reset_states()

            start = time.time()

            # Reduce the probability of teacher forcing each epoch
            _force_decay += (_force_decay * epoch)
            _force_prob = max(0, _force_prob - _force_decay)

            train_batches = 0
            val_batches = 0

            # Loop over batches
            for inp, targ in self.train_data.take(-1):

                # Reset the hidden state
                batch_size, out_len = targ.shape
                enc_hidden = self.encoder.initialize_hidden_state(batch_size)

                # Forward- and backpropagation
                loss, acc = self.train_step(inp, targ, _force_prob, enc_hidden, out_len)

                # Count the batch and print message
                train_batches += 1
                if train_batches % 5 == 0:
                    print((
                        f'Epoch {epoch + 1}:',
                        f'Loss {loss:.4f}, Acc {acc * 100:.2f}%',
                        f'after {train_batches} batches.'
                        ))

            # Save the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix='ocr-clean-ckpt')

            # Calculate validation loss and accuracy
            for inp, targ in self.val_data.take(-1):

                # Reset hidden state
                batch_size, out_len = targ.shape
                enc_hidden = self.encoder.initialize_hidden_state(batch_size)

                # Forward propagation
                val_loss_, val_acc_ = self.val_step(inp, targ, _force_prob,
                                                    enc_hidden, out_len)

                val_batches += 1

            print(f'\nEpoch {epoch + 1} Loss {loss:.2f}, Avg Acc {acc*100:.2f}%.')
            print(f'Tested on {len(self.seqs[2])} validation examples:')
            print(f'val_loss = {val_loss_:.2f} val_acc = {val_acc_*100:.2f}%')
            print(f'Time taken for 1 epoch: {format_time(time.time() - start)}')
            print('===========================\n')

    # Data pipeline
    def _load_saved_data(self, filename='saved-data.pickle'):
        """Load data from saved pickle

        Arguments:
        ==========
        filename(str): filename of the pickle"""

        with open(os.path.join(self.hparams.data_dir, filename), 'rb') as file:
            self.tkzr, self.seqs = p.load(file)

        self.train_data, self.val_data, self.tkzr, self.seqs, self.bucket_info = load_dataset(
            seqs=self.seqs,
            tkzr=self.tkzr,
            batch_size=self.hparams.max_batch_size,
            max_len=self.hparams.max_len,
            tolerance=self.hparams.tolerance)

        # Get vocab size
        self.hparams.num_chars = len(self.tkzr.word_index) + 1 # Add one for padding

    def _load_raw_data(self, x_path, y_path):
        """Loads data from provided csvs.

        The method presumes that you data is stored in two seperate csvs,
        one for the uncorrected texts, and one for the corrected texts.
        Each text occupies a single line of the csv.

        Arguments:
        ==========
        x_path (str): the path to the uncorrected csv
        y_path (str): the path to the corrected csv
        """
        self.train_data, self.val_data, self.tkzr, self.seqs, self.bucket_info = load_dataset(
            x_path=x_path,
            y_path=y_path,
            batch_size=self.hparams.max_batch_size,
            tolerance=self.hparams.tolerance,
            max_len=self.hparams.max_len)

        self.hparams.num_chars = len(self.tkzr.word_index)

        with open(
                os.path.join(
                    self.hparams.data_dir, 'saved_ocr_training_data.pickle'), 'wb'
                ) as file:
            p.dump((self.tkzr, self.seqs), file)

    def _define_model(self):
        """Defines model and creates checkpoints."""
        # Check that data has been imported and processed
        if self.hparams.num_chars is None:
            raise AttributeError('Model definition failed. Vocabulary size not known.')

        # If so, define model
        self.encoder = Encoder(**self.hparams.encoder())
        self.decoder = Decoder(**self.hparams.decoder())

        # Define checkpoints for saving
        self.checkpoint = tf.train.Checkpoint(optimizer=self.hparams.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

    def _load_saved_variables(self):
        """Load saved variables from checkpoint"""
        if self.hparams.checkpoint_dir is None:
            with path('cleanocr', 'saved_variables') as chk_dir:
                self.checkpoint.restore(chk_dir)
        else:
            self.checkpoint.restore(self.hparams.checkpoint_dir)

    def _create_checkpoint_dir(self):
        """Checks that checkpoint directory exists, creates it if it doesn't."""
        if self.hparams.checkpoint_dir is None:
            self.hparams.checkpoint_dir = os.path.join(os.getcwd(), 'model_checkpoints')
        if not os.path.exists(self.hparams.checkpoint_dir):
            print(f'Creating directory {self.hparams.checkpoint_dir}...')
            os.mkdir(self.hparams.checkpoint_dir)
        else:
            print(f'Model parameters will be saved to {self.hparams.checkpoint_dir}')
