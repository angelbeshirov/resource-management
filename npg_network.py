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
from jax.experimental.optimizers import optimizer, make_schedule

from jax import jit, ops
from jax import grad, value_and_grad
from jax import random

from parameters import Parameters
from data_generator import DataGenerator
from environment import ResourceManagementEnv
from logger import Logger, LogLevel

import pickle
import util

np.set_printoptions(threshold=maxsize)

class NPGNetwork:
    def __init__(self, parameters, env, logger = Logger(LogLevel['info']), plot_freq = 5):
        self.seed = 0
        self.number_episodes = parameters.number_episodes
        self.batch_size = parameters.batch_size
        self.episode_max_length = parameters.episode_max_length
        self.delta = 0.05
        self.F = None

        self.input_height = parameters.input_height_cmpct
        self.input_width = parameters.input_width_cmpct

        self.learning_rate = parameters.learning_rate
        self.eps = parameters.eps
        self.gamma = parameters.gamma

        self.env = env
        self.logger = logger
        self.plot_freq = plot_freq # on how many iterations plot is updated; 0 means no plotting

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

        self.opt_init, self.opt_update, self.get_params = self.npg()
            # optimizers.adam(step_size = self.learning_rate, b1 = self.b1, b2 = self.b2, eps = self.eps)

        self.opt_state = self.opt_init(self.initial_params)
        self.step = 0

        self.logger.info('Output shape of the model is {}.\n'.format(self.output_shape))

    def l2_regularizer(self, params, lmbda):
        """
        Define l2 regularizer: $\lambda \ sum_j ||theta_j||^2 $ for every parameter in the model $\theta_j$
        
        """
        return lmbda*jnp.sum(jnp.array([jnp.sum(jnp.abs(theta)**2) for theta in tree_flatten(params)[0] ]))

    def pseudo_loss(self, params, trajectory_batch):
        """
        Define the pseudo loss function for policy gradient. 
        
        params: object(jax pytree):
            parameters of the deep policy network.
        trajectory_batch: tuple (states, actions, returns) containing the RL states, actions and returns (not the rewards!): 
            states: np.array of size (batch_size, episode_max_length, input_height * input_width)
            actions: np.array of size (batch_size, episode_max_length)
            returns: np.array of size (batch_size, episode_max_length)
        
        Returns:
            -J_{pseudo}(\theta)

        """
        # extract data from the batch
        states, actions, returns = trajectory_batch
        # compute policy predictions
        preds = self.predict_jax(params, states)
        # combute the baseline
        baseline = jnp.mean(returns, axis=0)
        # select those values of the policy along the action trajectory
        preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=2), axis=2).squeeze()
        # return negative pseudo loss function (want to maximize reward with gradient Descent)
        return -jnp.sum(preds_select * (returns - baseline)) + self.l2_regularizer(params, 0.001)

    def get_grad_log(self, params, trajectory_batch):
        """
        Returns the gradient w.r.t to params of the log probability
        for trajectory batch
        NB: trajectory_batch is expected to have size (1, _, _) because only one example
        is being evaluated
        """
        state, action = trajectory_batch
        preds = self.predict_jax(params, state)
        preds_select = preds[0][action]
        return jnp.mean(preds_select)


    def update(self, batches):
        states = np.concatenate([b["states"] for b in batches])
        returns = np.concatenate([b["returns"] for b in batches])
        actions = np.concatenate([b["actions"] for b in batches])
        num_episodes = len(states)
        loss = 0

        for i in range(num_episodes):
            current_params = self.get_params(self.opt_state)

            self.calculate_fisher_matrices(current_params, states[i], actions[i], returns[i])
            l, g = value_and_grad(self.pseudo_loss)(current_params, \
                [states[i:i+1], returns[i:i+1], actions[i:i+1]])
            loss += l
            self.opt_state = self.opt_update(self.step, g, self.opt_state)
        loss /= float(num_episodes)
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
        trained_params = self.get_params(self.opt_state)
        file = open(path, 'wb')
        pickle.dump(trained_params, file, -1)
        file.close()

    def load(self, path):
        """
        Loads the model parameters from the specified file.
        """
        self.logger.info("Loading model from %s" % (path))
        file = open(path, 'rb')
        params = pickle.load(file)

        self.opt_state = self.opt_init(params)

    def calculate_fisher_matrices(self, current_params, states, actions, returns):
        """
        Computes Monte-Carlo approximation of Fisher information matrix
        by averaging over all the values of log_pi * log_pi^T
        """
        self.F = [None] * len(current_params)
        # Initialize Fisher matrices when relevant (2d layers)
        for i in range(len(current_params)):
            W = current_params[i][0] if len(current_params[i]) > 0 else None
            if W is not None and W.ndim == 2:
                d = len(W.flatten())
                self.F[i] = jnp.zeros((d, d), dtype=np.float)
        
        for t in range(len(states)):
            current_params = self.get_params(self.opt_state)
            # get log prob for current example
            grad_logp = grad(self.get_grad_log)(current_params, [states[t:t+1], actions[t:t+1]])

            for i in range(len(current_params)):
                if self.F[i] is not None:
                    glogp = grad_logp[i][0].flatten()
                    # add logprob gradient for current timestep
                    self.F[i] += jnp.outer(glogp, glogp)

        for i in range(len(current_params)):
            if self.F[i] is not None:
                self.F[i] /= float(len(states))

    @optimizer
    def npg(self, step_size = 0.001, delta=0.9, eps=1e-8):
        """Custom optimizer for Natural policy gradient
            performs update using the following rule
            theta <- theta + alpha * F^(-1) * theta'
            where F is the Fisher information matrix (self.F)
        Args:
            step_size: positive scalar, or a callable representing a step size schedule.

        Returns:
            An (init_fun, update_fun, get_params) triple.
        """
        step_size = make_schedule(step_size)
        def init(x0):
            return x0

        def update(i, g, theta):
            # check if FIM was computed
            if self.F is None:
                return theta - step_size(i) * g

            F = None
            g_flat = g.flatten()
            for f in self.F:
                # check if there is FIM for the current level
                # 1d levels such as Softmax, Relu, etc are not
                # supposed to have FIM computed
                if f is not None and f.shape[0] == len(g_flat):
                    F = f
            if F is None:
                # if no Fisher matrix is defined for current
                # layer use standard update
                return theta - step_size(i) * g

            # compute F_inverse using SVD approximation
            F_inv = jnp.linalg.pinv(F, rcond=(1e-6))
            natural_grad = F_inv.dot(g_flat)
            # g_nat = natural_grad.flatten()
            # alpha = jnp.sqrt((2 * delta) / jnp.dot(jnp.transpose(g_nat), jnp.dot(F, g_nat)))
            alpha = step_size(i)
            return theta - alpha * jnp.reshape(natural_grad, (theta.shape[0], theta.shape[1]))
                
        def get_params(x):
            return x
        return init, update, get_params