import torch

from keras import ops
from keras import optimizers
from keras.backend.torch.optimizers import torch_parallel_optimizer


class Adam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adam):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)
        local_step = ops.cast(self.iterations + 1, dtype)

        beta_1_power = ops.power(ops.cast(self.beta_1, dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, dtype), local_step)
        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        m_list = [
### Summary of Changes:
The code snippet in the file `keras/backend/torch/optimizers/torch_adam.py` needs to be modified to ensure that the `torch._foreach_mul_` and `torch._foreach_add_` functions are used correctly. The current code snippet shows the usage of these functions, but it seems that the parameters passed to the functions may not be correct. The necessary correction involves verifying and adjusting the parameters passed to the `torch._foreach_mul_` and `torch._foreach_add_` functions to ensure the correct operations are performed.
        torch._foreach_add_(
            v_list, torch._foreach_mul(grads, grads), alpha=1 - self.beta_2
        )

        if self.amsgrad:
            v_hat_list = [
                self._velocity_hats[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]
            torch._foreach_maximum_(v_hat_list, v_list)
            v_list = v_hat_list

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, alpha),
                torch._foreach_add(torch._foreach_sqrt(v_list), self.epsilon),
            ),
            alpha=-1,
        )
