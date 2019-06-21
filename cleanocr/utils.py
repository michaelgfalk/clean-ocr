"""Useful classes and functions"""

# pylint: disable=invalid-name;

import json
from collections import defaultdict
import tensorflow as tf

class HParams():
    """Hyperparameters for the model."""
    def __init__(self, **kwargs):

        # Handle missing arguments
        kwargs = defaultdict(lambda: None, kwargs)

        self.embedding_dim = kwargs['embedding_dim']
        self.K = kwargs['K'] # pylint: disable=invalid-name;
        self.data_dir = kwargs['data_dir']
        self.x_path = kwargs['x_path']
        self.y_path = kwargs['y_path']
        self.optimizer = tf.keras.optimizers.get(kwargs['optimizer'])
        self.max_batch_size = kwargs['max_batch_size']
        self.tolerance = kwargs['tolerance']
        self.max_len = kwargs['max_len']
        self.epochs = kwargs['epochs']
        self.checkpoint_dir = kwargs['checkpoint_dir']
        self.norm_lim = kwargs['norm_lim']
        self.units = kwargs['units']
        self.teacher_force_prob = kwargs['teacher_force_prob']
        self.teacher_force_decay = kwargs['teacher_force_decay']
        self.num_chars = kwargs['num_chars']

    @classmethod
    def from_json(cls, json_path): # pylint: disable=no-self-argument;
        """Constructs instance from provided json file."""
        with open(json_path, 'rt') as json_file:
            return cls(**json.load(json_file))

    def encoder(self):
        """Outputs keyword arguments for Encoder class."""
        return {'enc_units':self.units, 'embedding_dim':self.embedding_dim,
                'num_chars':self.num_chars, 'K':self.K}

    def decoder(self):
        """Outputs keyword arguments for Decoder class."""
        return{'dec_units':self.units, 'embedding_dim':self.embedding_dim,
               'num_chars':self.num_chars, 'K':self.K}


def format_time(flt):
    """Pretty prints the timestamp returned by time.time()"""

    h = flt//3600
    m = (flt % 3600)//60
    s = flt % 60
    out = []
    if h > 0:
        out.append(str(int(h)))
        if h == 1:
            out.append('hr,')
        else:
            out.append('hrs,')
    if m > 0:
        out.append(str(int(m)))
        if m == 1:
            out.append('min,')
        else:
            out.append('mins,')
    out.append(f'{s:.2f}')
    out.append('secs')
    return ' '.join(out)
