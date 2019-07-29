import tensorflow as tf
from tensorflow.keras import Model


class ASRN(Model):
    """ Attention-based Sequence Recognition Network
    """

    def __init__(self):
        super(ASRN, self).__init__()

    def call(self, inputs):
        return inputs

