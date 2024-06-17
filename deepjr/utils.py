
import os
import numpy as np
import tensorflow as tf
import random as python_random


def reset_random_seeds(seed_value=1234):
    ''' Set random seeds for reproducibility. '''
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    python_random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # Configure TensorFlow settings if necessary (rarely needed)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    # tf.config.threading.set_inter_op_parallelism_threads(1)
