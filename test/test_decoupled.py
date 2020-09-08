from test.utils import BotorchTestCase
from gpflow_sampling.samplers.decoupled_samplers import _decoupled_sampler_gpr
import tensorflow as tf
import gpflow
import itertools


class TestDecoupledSampler(BotorchTestCase):
    def test_decoupled_sampler(self):
        for train_n, dim, test_n, sample_shape, num_basis in itertools.product(
            [5, 20], [1, 3, 5], [3, 15], [[5, 3], [2], [3, 5, 7]], [1, 5]
        ):
            model = gpflow.models.GPR(
                data=(
                    tf.random.uniform(shape=[train_n, dim], dtype=tf.float64),
                    tf.random.uniform(shape=[train_n, 1], dtype=tf.float64),
                ),
                kernel=gpflow.kernels.Matern52(lengthscales=0.1, variance=1.0),
                mean_function=lambda X_data: 0.0,
                noise_variance=1e-3,
            )
            sampler = _decoupled_sampler_gpr(
                model=model,
                kernel=None,
                sample_shape=sample_shape,
                num_basis=6,
                dtype=tf.float64,
            )
            test_X = tf.random.uniform(shape=[test_n, dim], dtype=tf.float64)
            out = sampler(test_X)
            self.assertTupleEqual((*sample_shape, test_n, 1), tuple(out.shape))
