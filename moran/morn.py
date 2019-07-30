import numpy as np
import tensorflow as tf
from tensorflow.contrib.resampler import resampler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    RELU,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    UpSampling2D,
)


class MORN(Model):
    """ Multi-Object Rectification Network
    """

    def __init__(
        self, target_height: int, target_width: int, max_batch_size=256
    ):
        super(MORN, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.max_batch_size = max_batch_size
        self.filters = self._build_filters()
        self.pool = MaxPool2D(2, 1)
        self.grid = self._build_grid()
        self.grid_x = self.grid[:, :, :, 0].unsqueeze(3)
        self.grid_y = self.grid[:, :, :, 1].unsqueeze(3)

    def call(self, inputs, training=False):
        batch_size, height, width, channels = tf.shape(inputs)
        if training and np.random.random() > 0.5:
            return UpSampling2D(
                (self.target_height / height, self.target_width / width),
                interpolation="bilinear",
            )
        assert batch_size <= self.max_batch_size
        grid = self.grid[:batch_size]
        grid_x = self.grid_x[:batch_size]
        grid_y = self.grid_y[:batch_size]
        input_sample = UpSampling2D(
            (self.target_height / height, self.target_width / width),
            interpolation="bilinear",
        )

        offsets = self.filters(input_sample)
        offsets_pos = RELU()(offsets)
        offsets_neg = RELU()(-offsets)
        offsets_pool = self.pool(offsets_pos) - self.pool(offsets_neg)

        offsets_grid = tf.transpose(
            resampler(offsets_pool, grid), [0, 2, 3, 1]
        )
        input_offsets = tf.concat(
            values=[grid_x, grid_y + offsets_grid], axis=3
        )
        rectified_input = resampler(inputs, input_offsets)

        if training:
            return rectified_input

        offsets = self.filters(rectified_input)

        offsets_pos = RELU()(offsets)
        offsets_neg = RELU()(-offsets)
        offsets_pool = self.pool(offsets_pos) - self.pool(offsets_neg)

        offsets_grid += tf.transpose(
            resampler(offsets_pool, grid), [0, 2, 3, 1]
        )
        input_offsets = tf.concat(
            values=[grid_x, grid_y + offsets_grid], axis=3
        )
        rectified_input = resampler(inputs, input_offsets)

        return inputs

    def _build_filters(self):
        filters = Sequential(name="morn_filters")
        filters.add(MaxPool2D(2, 2))
        filters.add(Conv2D(64, 3, 1, 1))
        filters.add(BatchNormalization())
        filters.add(RELU())
        filters.add(MaxPool2D(2, 2))
        filters.add(Conv2D(128, 3, 1, 1))
        filters.add(BatchNormalization())
        filters.add(RELU())
        filters.add(MaxPool2D(2, 2))
        filters.add(Conv2D(64, 3, 1, 1))
        filters.add(BatchNormalization())
        filters.add(RELU())
        filters.add(Conv2D(16, 3, 1, 1))
        filters.add(BatchNormalization())

        return filters

    def _build_grid(self):
        y_points = (
            np.arange(self.target_height) * 2.0 / (self.target_height - 1) - 1
        )
        x_points = (
            np.arange(self.target_width) * 2.0 / (self.target_width - 1) - 1
        )

        grid = np.meshgrid(x_points, y_points, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.transpose(grid, (1, 0, 2))
        grid = np.expand_dims(grid, 0)
        grid = np.tile(grid, [self.max_batch_size, 1, 1, 1])
        return tf.Variable(grid)
