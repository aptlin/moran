import tensorflow as tf
from tensorflow.keras import Model


class Attention(Model):
    """ Attention-based Sequence Recognition Network
    """

    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs):
        return inputs

