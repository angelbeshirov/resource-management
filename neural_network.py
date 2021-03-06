import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

from sys import maxsize

from jax.experimental import stax 
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.tree_util import tree_flatten
from jax.experimental import optimizers # gradient descent optimizers

from jax import jit
from jax import grad, value_and_grad
from jax import random

from parameters import Parameters
from data_generator import DataGenerator
from environment import ResourceManagementEnv
from logger import Logger, LogLevel

import pickle
import util

np.set_printoptions(threshold=maxsize)

class Neural_network:
    def __init__(self, parameters, env, logger = Logger(LogLevel['info'])):
        self.seed = 0
        self.number_episodes = parameters.number_episodes
        self.batch_size = parameters.batch_size
        self.episode_max_length = parameters.episode_max_length

        self.input_height = parameters.input_height
        self.input_width = parameters.input_width

        self.learning_rate = parameters.learning_rate
        self.eps = parameters.eps
        self.gamma = parameters.gamma

        self.env = env
        self.logger = logger

        rng = random.PRNGKey(self.seed)

        self.initialize_params, self.predict_jax = stax.serial(
                                            Dense(20),
                                            Relu,
                                            Dense(parameters.network_output_dim),
                                            LogSoftmax
        )
        
        # process simultaneously all time steps and MC samples, generated in a single training iteration
        self.input_shape = (-1, self.episode_max_length, self.input_height * self.input_width)
        self.output_shape, self.initial_params = self.initialize_params(rng, self.input_shape)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size = self.learning_rate)
            # optimizers.rmsprop(step_size = self.learning_rate, gamma = self.gamma, eps = self.eps)
            # optimizers.adam(step_size = self.learning_rate)

        self.opt_state = self.opt_init(self.initial_params)
        self.step = 0

        self.logger.info('Output shape of the model is {}.\n'.format(self.output_shape))

    def l2_regularizer(self, params, lmbda):
        """
        Define l2 regularizer: $\lambda \ sum_j ||theta_j||^2 $ for every parameter in the model $\theta_j$
        
        """
        return lmbda*jnp.sum(jnp.array([jnp.sum(jnp.abs(theta)**2) for theta in tree_flatten(params)[0] ]))

    def pseudo_loss(self, params, batches):
        """
        Define the pseudo loss function for policy gradient.
        """
        loss = 0
        for batch in batches:
            states = batch["states"]
            actions = batch["actions"]
            returns = batch["returns"]

            preds = self.predict_jax(params, states)

            baseline = jnp.mean(returns, axis=0)
            preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=2), axis=2).squeeze()
            loss += (-jnp.mean(jnp.sum(preds_select * (returns - baseline))))

        return loss + self.l2_regularizer(params, 0.001) # try to divide by len(batches)?

    def update(self, batches):
        """
        batch: np.array
            batches containing the data used to update the model - N batches for each job sequence simulation
            the parameters are updated for each simulation together
            
        Returns: loss
        """
        current_params = self.get_params(self.opt_state)
        loss, grad_params = value_and_grad(self.pseudo_loss)(current_params, batches)

        self.opt_state = self.opt_update(self.step, grad_params, self.opt_state)
        self.step = self.step + 1
        return loss

    def predict(self, data):
        """
        Predict an action based on the data parameter.
        """
        params = self.get_params(self.opt_state)
        return self.predict_jax(params, data)
        
    def save(self, path):
        """
        Saves the model parameters into the specified file.
        """
        file = open(path, 'wb')
        pickle.dump(optimizers.unpack_optimizer_state(self.opt_state), file, -1)
        file.close()

    def load(self, path):
        """
        Loads the model parameters from the specified file.
        """
        file = open(path, 'rb')
        state = pickle.load(file)

        self.opt_state = optimizers.pack_optimizer_state(state)