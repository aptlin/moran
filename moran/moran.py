import tensorflow as tf
from tensorflow.keras import Model


class MORAN(Model):
    """ Multi-Object Rectified Attention Network 	
    """

    def __init__(self):
        super(MORAN, self).__init__()

    def call(self, inputs):
        return inputs
