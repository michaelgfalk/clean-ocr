"""Defines training loop for model.
"""

import tensorflow as tf

from cleanocr.preprocess import load_dataset
from cleanocr.model import Encoder,Decoder
from cleanocr.utils import format_time

import os
import time
import pickle as p

class Trainer():
    """Class for orgainising training loop."""
    def __init__(
        self,
        data_dir,
        optimizer = 'Adam',
        max_batch_size = 16,
        tolerance = 1.2,
        max_len = 180,
        epochs = 10,
        K = 32,
        checkpoint_dir = None,
        norm_lim = 3,
        embedding_dim = 25,
        units = 50,
        teacher_force_prob = 0.99,
        teacher_force_decay = 0.005
        ):

        # Hyperparameters
        self.data_dir = data_dir,
        self.optimizer = tf.keras.optimizers.get(optimizer),
        self.max_batch_size = max_batch_size,
        self.tolerance = tolerance,
        self.max_len = max_len,
        self.epochs = epochs,
        self.K = K,
        self.checkpoint_dir = checkpoint_dir,
        self.norm_lim = norm_lim,
        self.embedding_dim = embedding_dim,
        self.units = units,
        self.teacher_force_prob = teacher_force_prob,
        self.teacher_force_decay = teacher_force_decay

        # Loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Metrics 
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

    # Functions for training loop
    def loss_function(self, real, pred):
        # Mask: ignore the model's predictions where the ground truth is padding
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        
        # Calculate the loss
        loss_ = self.loss_object(real, pred)

        # Make mask compatible with the loss output
        mask = tf.cast(mask, dtype=loss_.dtype)
        
        # Multiply the losses by the mask (i.e. zero out all losses where there's just padding)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)

    def sample(self, predictions, target, teacher_force_prob, t, batch_size):
        """Samples from previous timestep with given probability.
        
        Arguments:
        ==========
        predictions (tensor): [m x n] tensor of output from previous timestep
        target (tensor): [m x t x n] tensor of target sequences
        teacher_force_prob (tensor): float, probability of using target letter as
            next input
        t (int): current timestep
        batch_size (int): the size of the current batch
        
        Returns:
        ==========
        dec_input (tensor): [m x 1] tensor of mixed target tokens and sampled ones
        """
        
        # NB: tf.random.categorical works with log probabilities
        
        # Sample next input from predictions
        samples = tf.random.categorical(predictions, num_samples = 1, dtype = tf.int32)
        
        # Get gold standard input from target sequence
        teachers = tf.expand_dims(target[:, t], 1)
        
        # Create mask using teacher_force_prob
        # Copy the probabilities m times
        mask = tf.tile([tf.math.log(1-teacher_force_prob), tf.math.log(teacher_force_prob)], [batch_size])
        # Reshape into an m x 2 tensor
        mask = tf.reshape(mask, [batch_size, 2])
        # Sample to create mask
        mask = tf.random.categorical(mask, num_samples = 1)
        # Recast into a boolean tensor
        mask = tf.cast(mask, dtype = tf.bool)
        
        # Update samples with true value where mask says to do so
        dec_input = tf.where(mask, teachers, samples)
        
        return dec_input

    @tf.function
    def train_step(self, inp, targ, teacher_force_prob, enc_hidden, batch_size, out_len):
        loss = 0
                
        with tf.GradientTape() as tape:
            
            # Encode input sequence: retrieve last hidden state and attention matrix
            enc_hidden, C = self.encoder(inp, enc_hidden)

            # Set initial state of decoder to same as encoder
            dec_hidden = enc_hidden
            
            # Set initial input of decoder to initial input of input sequence
            dec_input = tf.expand_dims(inp[:,0], 1)
            
            for t in range(1, out_len):
                # passing enc_output to the decoder
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden, C)

                loss += self.loss_function(targ[:, t], predictions)
                self.update_accuracy(targ[:, t], predictions, self.train_acc)

                # Teacher forcing/scheduled sampling
                dec_input = self.sample(predictions, targ, teacher_force_prob, t, batch_size)

        self.train_loss.update_state(loss)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)
        
        # Clip gradients
        clipped_gradients = [tf.clip_by_norm(grad, self.norm_lim) for grad in gradients]

        self.optimizer.apply_gradients(zip(clipped_gradients, variables))
        
        return self.train_loss.result(), self.train_acc.result()

    @tf.function
    def val_step(self, inp, targ, teacher_force_prob, enc_hidden, batch_size, out_len):
    
        loss = 0
        
        # Begin feeding data to network
        enc_hidden, C = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(inp[:,0], 1)
        
        # Cycle through the rest of the time steps
        for t in range(1, out_len):
            # Pass enc_output to the decoder
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden, C)
            
            # Calculate loss and acc
            loss += self.loss_function(targ[:,t], predictions)
            self.update_accuracy(targ[:, t], predictions, self.val_acc)
            
            # Use teacher forcing or sampling as appropriate...
            dec_input = self.sample(predictions, targ, teacher_force_prob, t, batch_size)
            
        # Calculate val_loss
        self.val_loss.update_state(loss)
        
        return self.val_loss.result(), self.val_acc.result()

    @staticmethod
    def update_accuracy(real, pred, acc_object):
        """Updates accuracy metric, ignoring padding variable."""
    
        # Find padding
        mask = tf.math.greater(real, [0]) # 'True' for all non-zero values
        mask = tf.cast(mask, dtype = tf.dtypes.int8) # 'True'  = 1, 'False' = 0

        # Compute accuracy
        acc_object.update_state(real, pred, sample_weight = mask)

    # Data pipeline
    def load_saved_data(self, filename):
        """Load data from saved pickle
        
        Arguments:
        ==========
        filename(str): filename of the pickle"""

        with open(os.path.join(self.data_dir, filename), 'rb') as file:
                self.tkzr, self.seqs = p.load(file)

        self.train_data, self.val_data, self.tkzr, self.seqs, self.bucket_info = load_dataset(
            seqs = self.seqs,
            tkzr = self.tkzr,
            batch_size = self.max_batch_size,
            max_len = self.max_len,
            tolerance = self.tolerance)

        # Get vocab size
        self.num_chars = len(self.tkzr.word_index) + 1 # Add one for padding

    def load_raw_data(self, x_path, y_path):
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
            x_path = x_path,
            y_path = y_path,
            batch_size = self.max_batch_size,
            tolerance = self.tolerance,
            max_len = self.max_len)

        self.num_chars = len(self.tkzr.word_index)

        with open(os.path.join(self.data_dir, 'saved_ocr_training_data.pickle'), 'wb') as file:
            p.dump((self.tkzr, self.seqs), file)

    def define_model(self):
        """Defines model and creates checkpoints."""
        # Check that data has been imported and processed
        try:
            assert hasattr(self, 'num_chars')
        except AssertionError:
            raise AttributeError('Model definition failed. Vocabulary size not known.') from AssertionError

        # If so, define model
        self.encoder = Encoder(self.num_chars, self.embedding_dim, self.units, self.K)
        self.decoder = Decoder(self.num_chars, self.embedding_dim, self.units, self.K)

        # Define checkpoints for saving
        self.checkpoint_dir = os.path.join(self.data_dir, 'checkpoints_sentences')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoder,
                                        decoder=self.decoder)

    def load_saved_variables(self):
        """Load saved variables from checkpoint"""
        pass

    def train(self):

        # Teacher forcing hparams
        _force_decay = self.teacher_force_decay
        _force_prob = self.teacher_force_prob

        # Loop over epochs
        for epoch in range(self.epochs):
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
                loss, acc = self.train_step(inp, targ, _force_prob, enc_hidden, batch_size, out_len)
                
                # Count the batch and print message
                train_batches += 1
                if train_batches % 5 == 0:
                    print(f'Epoch {epoch + 1}: Loss {loss:.4f}, Acc {acc * 100:.2f}% after {train_batches} batches')
            
            # Save the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                
            # Calculate validation loss and accuracy
            for inp, targ in self.val_data.take(-1):

                # Reset hidden state
                batch_size, out_len = targ.shape
                enc_hidden = self.encoder.initialize_hidden_state(batch_size)

                # Forward propagation
                val_loss_, val_acc_ = self.val_step(inp, targ, _force_prob, enc_hidden, batch_size, out_len)

                val_batches += 1

            print(f'\nEpoch {epoch + 1} Loss {loss:.2f}, Avg Acc {acc*100:.2f}%.')
            print(f'Tested on {len(self.seqs[2])} validation examples:')
            print(f'val_loss = {val_loss_:.2f} val_acc = {val_acc_*100:.2f}%')
            print(f'Time taken for 1 epoch: {format_time(time.time() - start)}\n===========================\n\n')