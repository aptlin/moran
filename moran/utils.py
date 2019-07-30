import tensorflow as tf


def grid_sample_2d(image, grid):
    """ Sample image guided by the given grid.

    The implementation is adapted from
    https://stackoverflow.com/a/52896615/3581829

    :type tf.Tensor4D:
    :param image: The image to sample from

    :type numpy.meshgrid:
    :param grid: The sampling grid

    :rtype: The sampled tensor, having the same shape as `image`
    """
    image_shape = tf.shape(image)
    image_height = image_shape[1]
    image_width = image_shape[2]

    # Find interpolation sides
    i, j = grid[..., 0], grid[..., 1]
    i = tf.cast(image_height - 1, grid.dtype) * (i + 1) / 2
    j = tf.cast(image_width - 1, grid.dtype) * (j + 1) / 2
    i_1 = tf.maximum(tf.cast(tf.floor(i), tf.int32), 0)
    i_2 = tf.minimum(i_1 + 1, image_height - 1)
    j_1 = tf.maximum(tf.cast(tf.floor(j), tf.int32), 0)
    j_2 = tf.minimum(j_1 + 1, image_width - 1)
    # Gather pixel values
    n_idx = tf.tile(
        tf.range(image_shape[0])[:, tf.newaxis, tf.newaxis],
        tf.concat([[1], tf.shape(i)[1:]], axis=0),
    )
    q_11 = tf.gather_nd(image, tf.stack([n_idx, i_1, j_1], axis=-1))
    q_12 = tf.gather_nd(image, tf.stack([n_idx, i_1, j_2], axis=-1))
    q_21 = tf.gather_nd(image, tf.stack([n_idx, i_2, j_1], axis=-1))
    q_22 = tf.gather_nd(image, tf.stack([n_idx, i_2, j_2], axis=-1))
    # Interpolation coefficients
    di = tf.cast(i, image.dtype) - tf.cast(i_1, image.dtype)
    di = tf.expand_dims(di, -1)
    dj = tf.cast(j, image.dtype) - tf.cast(j_1, image.dtype)
    dj = tf.expand_dims(dj, -1)
    # Compute interpolations
    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj
    return q_ij
