from keras import ops
from keras.api_export import keras_export
from keras.layers.input_spec import InputSpec
from keras.layers.layer import Layer


@keras_export("keras.layers.UpSampling1D")
class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs.

    Repeats each temporal step `size` times along the time axis.
### Summary of Changes:
The code snippet in the file `keras/layers/reshaping/up_sampling1d.py` needs to be modified to fix the formatting issue in the example output comments. The current example output comments have incorrect formatting with missing spaces and newlines. The correction involves adjusting the formatting in the example output comments to ensure clarity and readability.
        size: Integer. Upsampling factor.

    Input shape:
        3D tensor with shape: `(batch_size, steps, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, upsampled_steps, features)`.
    """

    def __init__(self, size=2, **kwargs):
        super().__init__(**kwargs)
        self.size = int(size)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        size = (
            self.size * input_shape[1] if input_shape[1] is not None else None
        )
        return [input_shape[0], size, input_shape[2]]

    def call(self, inputs):
        return ops.repeat(x=inputs, repeats=self.size, axis=1)

    def get_config(self):
        config = {"size": self.size}
        base_config = super().get_config()
        return {**base_config, **config}
