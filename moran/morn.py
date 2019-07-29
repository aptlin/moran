import tensorflow as tf
from tensorflow.keras import Model


class MORN(Model):
    """ Multi-Object Rectification Network 	
    """

    def __init__(self):
        super(MORN, self).__init__()

    def call(self, inputs):
        return inputs

