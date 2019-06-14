"""Functions for ingesting training and test data
"""

# pylint: disable=invalid-name;

import csv
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split

def import_raw_data(source, target):
    """Imports articles from csv and packages for training."""

    # Lists for import
    raw_x = []
    raw_y = []

    # Import raw text
    with open(source, "rt") as f:
        x_reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in x_reader:
            raw_x.append(row[1])
    with open(target, "rt") as f:
        y_reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in y_reader:
            raw_y.append(row[1])

    # Drop header rows
    raw_x = raw_x[1:]
    raw_y = raw_y[1:]

    return raw_x, raw_y

def split_sentences(x_data, y_data, tolerance=10, adjust=10.,
                    start='स', end='ए', max_len=200):
    """
    Splits articles into sentences.

    The function does some light preprocessing, and tries as best as possible to align the sentences
    in the source and target documents. It uses the target documents as a guide to where the
    sentence boundaries are. It finds the full stops in the target documents, the looks for the
    nearest likely sentence-boundary character in the source sentences.

    It also performs some light preprocessing. Strings of junky puncutation are deleted from the
    source stings, and the supplied 'start' and 'end' characters are appended to each sentence.

    Arguments:
    ==========
    x_data:    a list of strings, each representing a single uncorrected OCR document
    y_data:    a list of strings, representing the corresponding corrected texts
    tolerance: an integer, how far either side of the y full stop to look for
              the corresponding x full stop
    adjust:    a float, how much to adjust the tolerance to account for differing lengths
              of x and y
    start,end: special characters to append to the start and end of each sentence
    max_len:   the maximum allowable length of a sentence, in characters. Sentences longer than
              this will be truncated.

    Returns:
    ==========
    idxs: a list of tuples, giving the article number and sentence number for each
    sentence
    x_sentences: a list of strings, the uncorrected sentences
    y_sentences: a list of strings, the corrected sentences
    """

    # Define key variables
    idxs = []
    x_sentences = []
    y_sentences = []
    article_num = -1
    punct = {',', ';', ':', ' '}
    clean_rgx = re.compile(r'\W{4,}')
    # Find a full stop that doesn't have Dr, Mr, Mrs or an initial in front
    search_rgx = re.compile(r'(?<!Mr)(?<!Mrs)(?<!Dr)(?<!\b[A-Z])\.')

    for x, y in zip(x_data, y_data):

        # Clean out junky punctuation
        x = clean_rgx.sub(' ', x, count=0) # count = 0 --> replace all occurences

        # Adjust the tolerance based on the length of the sentences
        len_diff = len(y) - len(x)
        # If y is longer than x, len_diff will be positive, meaning
        # that lo_tol will increase, and hi_tol will decrease, shifting the
        # search_area to the left of x
        # If x is longer than y, len_diff will be negative, meaning
        # that lo_tol will decrease, and hi_tol will increase, shifting the
        # search_area to the right of x
        lo_tol = tolerance + round(len_diff / len(y) * adjust)
        hi_tol = tolerance - round(len_diff / len(y) * adjust)

        article_num += 1
        sentence_num = -1

        # Loop over target document:
        while y:
            # Increment sentence counter
            sentence_num += 1

            # Find the next sentence boundary
            y_stop_mtch = search_rgx.search(y)

            # If a full stop is found, try to align the sentences
            if y_stop_mtch:

                # The span method of the returned match object give the start and
                # end idx of the match
                y_stop_idx = y_stop_mtch.span()[1]

                # Pop out the next sentence
                next_y_sent, y = y[:y_stop_idx], y[y_stop_idx:]

                # See if there is a full stop near where the full stop is in y
                # Find the nearest full stop to the y full stop and split there:
                lo_bnd, hi_bnd = y_stop_idx - lo_tol, y_stop_idx + hi_tol
                search_area = x[lo_bnd:hi_bnd]

                # Look for a corresponding full stop in x
                x_match = search_rgx.search(x, lo_bnd, hi_bnd)

                # If a full stop is found, split on it
                # NB: regex.search will find the first full stop in the search area
                if x_match:
                    x_stop_idx = x_match.span()[1]
                    next_x_sent, x = x[:x_stop_idx], x[x_stop_idx:]

                # If not, check for other likely punctuation...
                # NB: You can use set operations on strings!
                elif punct.intersection(search_area):
                    # Get the positions of the possible candidates
                    x_stops = [pos for pos, char in enumerate(search_area) if char in punct]
                    # Find the candidate in the most likely position (near the middle)
                    # NB: Deduct the low tolerance to align seach_area with x
                    x_stop_idx = x_stops[round(len(x_stops)/2)] + y_stop_idx - lo_tol + 1
                    # Partition x
                    next_x_sent, x = x[:x_stop_idx], x[x_stop_idx:]

                # If there is no likely splitting point, just split on y_stop_idx
                else:
                    next_x_sent, x = x[:y_stop_idx], x[y_stop_idx:]

            # If there are no full stops left, go to end of document
            else:
                next_y_sent, y = y, ''
                next_x_sent, y = x, ''

            # Final cleaning. Drop any ridiculously short sentences
            if len(next_x_sent) > max_len or len(next_y_sent) > max_len:
                next_x_sent = next_x_sent[:max_len]
                next_y_sent = next_y_sent[:max_len]

            # End iteration if the sentences are too short (likely to be junk)
            if len(next_x_sent) < 10 and len(next_y_sent) < 10:
                continue

            # Append special characters
            next_x_sent = start + next_x_sent.strip() + end
            next_y_sent = start + next_y_sent.strip() + end

            x_sentences.append(next_x_sent)
            y_sentences.append(next_y_sent)
            idxs.append((article_num, sentence_num))

        if (article_num + 1) % 1000 == 0:
            print(f'Article {article_num + 1} of {len(x_data)} parsed.')

    return x_sentences, y_sentences, idxs

def tokenize(x, y, tkzr=None):
    """Instantiates tokenizer if not passed, fits and applies."""

    if tkzr is None:
        # Fit tokenizer
        tkzr = tf.keras.preprocessing.text.Tokenizer(
            num_words=None,
            filters=None,
            lower=False,
            char_level=True
        )
        tkzr.fit_on_texts(x + y)

    # Apply to texts
    x = tkzr.texts_to_sequences(x)
    y = tkzr.texts_to_sequences(y)

    return x, y, tkzr

def t_value(x_tensor, y_tensor):
    """Returns the maximum length of x and y."""

    x_len = tf.shape(x_tensor)[0]
    y_len = tf.shape(y_tensor)[0]
    lengths = tf.stack([x_len, y_len], axis=0)

    return tf.reduce_max(lengths)

def bucket_fun(x, y, max_batch_size, tolerance=1.2):
    """Returns a bucketing function that can be applied to a tf.Dataset.

    Arguments:
    ==========
    x (list): the x sequences
    y (list): the target sequences
    max_batch_size (int): the maximum allowed batch size of the buckets
    tolerance (float): the maximum allowed ratio of longest to shortest sequence in a batch

    Returns:
    ==========
    bucketor (fun): instance of tf.data.experimental.bucket_by_sequence_length"""

    # Sort x and y in ascending order of length
    x_y = sorted(zip(x, y), key=lambda tup: max(len(tup[0]), len(tup[1])))

    # Iterate and accumulate boundaries and sizes
    bucket_boundaries, bucket_batch_sizes = [], []
    ex_num, min_len = 0, max(len(x_y[0][0]), len(x_y[0][1]))
    for x, y in x_y:

        # What is the length of this example?
        this_len = max(len(x), len(y)) + 1

        # If length diff is greater than the tolerance,
        # Or if maximum batch_size is reached
        if this_len > (min_len * tolerance) or ex_num == max_batch_size:
            # Mark the end of this batch with the previous example's
            # length and ex_num:
            bucket_boundaries.append(bucket_len)
            bucket_batch_sizes.append(ex_num)
            # Begin a new batch
            ex_num = 1
            min_len = this_len
            bucket_len = this_len

        # Otherwise, just keep iterating through
        else:
            bucket_len = this_len
            ex_num += 1

    # Add the batch_size of the final bucket
    bucket_batch_sizes.append(ex_num)
    # Update final bucket_boundary so it can accomodate the longest sequence
    bucket_boundaries[-1] = this_len

    bucketor = tf.data.experimental.bucket_by_sequence_length(
        element_length_func=t_value,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes
    )

    return bucketor, bucket_boundaries, bucket_batch_sizes

def load_dataset(x_path=None, y_path=None, batch_size=128, tolerance=1.2,
                 test_size=.2, seqs=None, tkzr=None, max_len=None):
    """Load the dataset either from a saved tuple or from the raw text"""

    if seqs is None and tkzr is None:
        # Load from raw text files
        x, y = import_raw_data(x_path, y_path)
        x, y, idxs = split_sentences(x, y, max_len=max_len)
        x, y, tkzr = tokenize(x, y)
        # seqs == (x_train, x_test, y_train, y_test, idx_train, idx_test)
        seqs = train_test_split(x, y, idxs, test_size=test_size)

    # Convert training data to dataset and bucket
    output_dtypes = (tf.int32, tf.int32)
    output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))

    train_gen = lambda: ((x, y) for x, y in zip(seqs[0], seqs[2]))
    train_data = tf.data.Dataset.from_generator(train_gen, output_dtypes, output_shapes)

    train_bucketor, train_bnds, train_btchs = bucket_fun(seqs[0], seqs[2], batch_size, tolerance)
    train_data = train_data.apply(train_bucketor)

    # Flip inputs on the time dimension
    def flip(inp, tar):
        flipped = tf.reverse(inp, [1])
        return flipped, tar

    train_data = train_data.map(flip)

    # Same for test
    test_gen = lambda: ((x, y) for x, y in zip(seqs[1], seqs[3]))
    test_data = tf.data.Dataset.from_generator(test_gen, output_dtypes, output_shapes)

    test_bucketor, test_bnds, test_btchs = bucket_fun(seqs[1], seqs[3], batch_size, tolerance)
    test_data = test_data.apply(test_bucketor)

    test_data = test_data.map(flip)

    bucket_info = (train_bnds, train_btchs, test_bnds, test_btchs)

    return train_data, test_data, tkzr, seqs, bucket_info
