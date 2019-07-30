import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.contrib.resampler import resampler
from tensorflow.keras import Model


class FractionalPickup(Model):
    """ Fractionally picks up the neighboring features in the training phase.
    """

    def __init__(self):
        super(FractionalPickup, self).__init__()

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        assert len(inputs_shape) == 4
        assert inputs_shape[2] == 1

        height = inputs_shape[2]
        width = inputs_shape[3]
        widths = np.arange(width) * 2.0 / (width - 1) - 1
        idx = int(npr.rand() * len(widths))
        if idx > 0 and idx < width - 1:
            beta = npr.rand() / 4.0
            previous = beta * widths[idx] + (1 - beta) * widths[idx - 1]
            current = beta * widths[idx - 1] + (1 - beta) * widths[idx]
            widths[idx - 1] = previous
            widths[idx] = current
        grid = np.meshgrid(widths, height, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.transpose(grid, (1, 0, 2))
        grid = np.expand_dims(grid, 0)
        grid = np.tile(grid, [inputs_shape[0], 1, 1, 1])
        self.grid = tf.Variable(grid)
        inputs_offset = resampler(inputs, self.grid)

        return inputs_offset
