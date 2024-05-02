from keras import ops
from keras.api_export import keras_export
from keras.layers.input_spec import InputSpec
from keras.layers.layer import Layer
from keras.utils import argument_validation


@keras_export("keras.layers.ZeroPadding1D")
class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

import numpy as np
import keras

class ZeroPadding1D(keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        self.padding = padding
        super(ZeroPadding1D, self).__init__(**kwargs)

    def call(self, inputs):
        input_shape = inputs.shape
        padded_shape = (input_shape[0], input_shape[1] + 2 * self.padding, input_shape[2])
        padded_inputs = np.zeros(padded_shape)
        padded_inputs[:, self.padding: self.padding + input_shape[1], :] = inputs
        return padded_inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding, input_shape[2])
    Args:
        padding: Int, or tuple of int (length 2), or dictionary.
            - If int: how many zeros to add at the beginning and end of
              the padding dimension (axis 1).
            - If tuple of 2 ints: how many zeros to add at the beginning and the
              end of the padding dimension (`(left_pad, right_pad)`).

    Input shape:
        3D tensor with shape `(batch_size, axis_to_pad, features)`

    Output shape:
        3D tensor with shape `(batch_size, padded_axis, features)`
    """

    def __init__(self, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = argument_validation.standardize_tuple(
            padding, 2, "padding", allow_zero=True
        )
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if output_shape[1] is not None:
            output_shape[1] += self.padding[0] + self.padding[1]
        return tuple(output_shape)

    def call(self, inputs):
        all_dims_padding = ((0, 0), self.padding, (0, 0))
        return ops.pad(inputs, all_dims_padding)

    def get_config(self):
        config = {"padding": self.padding}
        base_config = super().get_config()
        return {**base_config, **config}
