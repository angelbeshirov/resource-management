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

class Neural_network:
    def __init__(self, parameters, env, logger = Logger(LogLevel['info']), plot_freq = 5):
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
        self.output_shape, self.inital_params = self.initialize_params(rng, self.input_shape)

        self.opt_init, self.opt_update, self.get_params = \
            optimizers.rmsprop(step_size = self.learning_rate, gamma = self.gamma, eps = self.eps)
            # optimizers.adam(step_size = self.learning_rate, b1 = self.b1, b2 = self.b2, eps = self.eps)

        self.opt_state = self.opt_init(self.inital_params)
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
        return -jnp.mean(jnp.sum(preds_select * (returns - baseline) )) + self.l2_regularizer(params, 0.001)

    def update(self, batch):
        """
        Updates the neural network parameters.

        batch: np.array
            batch containing the data used to update the model
        """

        # get current parameters of the model
        current_params = self.get_params(self.opt_state)

        # compute gradients
        loss, grad_params = value_and_grad(self.pseudo_loss)(current_params, batch)

        # perform the update
        self.opt_state = self.opt_update(self.step, grad_params, self.opt_state)

        # increment the step
        self.step = self.step + 1
        return loss

    def predict(self, data):
        """
        Predict an action based on the data parameter.
        """
        params = self.get_params(self.opt_state)
        return self.predict_jax(params, data)

    # Pseudocode for training algorithm
    # for each iteration:
    #   ∆θ ← 0
    #   for each jobset:
    #       run episode i = 1, . . . , N :
    #       {s_i0, a_i0, r_i1, ... s_iLi, aiLi, riLi+1}
    #       compute returns: v_it = Sum[s=t to Li](gama^(s-t)*r_is)
    #       for t = 1 to L:
    #       compute baseline: b_t = 1/N*Sum[i=1 to N](v_it)
    #       for i = 1 to N :
    #           ∆θ ← ∆θ + α∇_θ log π_θ(s_it , a_it )(v_it − b_it )
    #       end
    #   end
    # end
    # θ ← θ + ∆θ % batch parameter update
    def train(self):
        best_reward = -maxsize

        # preallocate data using arrays initialized with zeros
        state=np.zeros((self.input_height * self.input_width,), dtype=np.float32)
    
        states = np.zeros((self.batch_size, self.episode_max_length, self.input_height * self.input_width), dtype=np.float32)
        actions = np.zeros((self.batch_size, self.episode_max_length), dtype=np.int)
        returns = np.zeros((self.batch_size, self.episode_max_length), dtype=np.float32)
    
        # mean reward at the end of the episode
        mean_final_reward = np.zeros(self.number_episodes, dtype=np.float32)
        # total reward at the end of the episode
        total_final_reward = np.zeros_like(mean_final_reward)
        # standard deviation of the reward at the end of the episode
        std_final_reward = np.zeros_like(mean_final_reward)
        # batch minimum at the end of the episode
        min_final_reward = np.zeros_like(mean_final_reward)
        # batch maximum at the end of the episode
        max_final_reward = np.zeros_like(mean_final_reward)
        mean_avg_slowdowns = np.zeros(self.number_episodes, dtype=np.float32)
        mean_avg_completion_times = np.zeros(self.number_episodes, dtype=np.float32)

        self.logger.info("\nStarting training...\n")

        # set the initial model parameters in the optimizer
        self.opt_state = self.opt_init(self.inital_params)

        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        losses = []

        # loop over the number of training episodes
        for episode in range(self.number_episodes): 
            
            ### record time
            start_time = time.time()
            
            # get current policy  network params
            current_params = self.get_params(self.opt_state)

            avg_slowdowns = np.zeros(self.batch_size, dtype=np.float32)
            avg_completion_times = np.zeros(self.batch_size, dtype=np.float32)
            
            # MC sample
            for j in range(self.batch_size):
                
                self.logger.debug("Monte carlo simulation %d started!" % (j))
                # reset environment to the initial state
                self.env.reset()
            
                # zero rewards array (auxiliary array to store the rewards, and help compute the returns)
                rewards = np.zeros((self.episode_max_length, ), dtype=np.float32)

                # loop over steps in an episode
                for time_step in range(self.episode_max_length):

                    # select state
                    state = self.env.retrieve_state().reshape(1, -1) # Fix the shape?
                    states[j, time_step, :] = state

                    # select an action according to current policy
                    pi_s = np.exp(self.predict_jax(current_params, state))
                    action = np.random.choice(self.env.actions, p = pi_s[0])
                    actions[j,time_step] = action # batch_sizexepisode_max_length

                    # take an environment step
                    _ , reward, done = self.env.step(action)

                    # If everything is executed the rest of the rewards will be 0
                    # which is exactly the expected behaviur since the environment
                    # only returns negative rewards (-1/T_j)
                    if done:
                        #env.log("No more jobs in the environment, everything is executed.")
                        self.logger.info("No more jobs in the environment, everything is executed.")
                        break
                    # state[:] = s.reshape([1, -1])

                    # store reward
                    rewards[time_step] = reward
                    
                # compute reward-to-go 
                # e.g. rewards = 1 2 3 4
                # rewards[::-1] = 4 3 2 1
                # cumsum -> 4 7 9 10
                # [::-1] -> 10 9 7 4
                returns[j,:] = jnp.cumsum(rewards[::-1])[::-1]
                avg_slowdowns[j] = self.env.get_average_slowdown()
                avg_completion_times[j] = self.env.get_average_completion_time()
                    
            # define batch of data
            trajectory_batch = (states, actions, returns)
            
            # update model
            loss = self.update(trajectory_batch)
            losses.append(loss)
                    
            ### record time needed for a single epoch
            episode_time = time.time() - start_time
            
            # check performance
            mean_final_reward[episode]=jnp.mean(returns[:,-1])
            total_final_reward[episode] = jnp.sum(returns[:,-1])
            std_final_reward[episode] =jnp.std(returns[:,-1])
            min_final_reward[episode], max_final_reward[episode] = np.min(returns[:,-1]), np.max(returns[:,-1])
            mean_avg_slowdowns[episode] = jnp.mean(avg_slowdowns)
            mean_avg_completion_times[episode] = jnp.mean(avg_completion_times)

            if total_final_reward[episode] >= best_reward:
                self.logger.info("Saving new best model with reward %d in episode %d" % (total_final_reward[episode], episode))
                self.save("./best_model.pkl", self.env)
                best_reward = total_final_reward[episode]

            # print results every 10 epochs
            #if episode % 5 == 0:
            self.logger.info("episode {} in {:0.2f} sec".format(episode, episode_time))
            self.logger.info("total reward: {:0.4f}".format(total_final_reward[episode]))
            self.logger.info("mean reward: {:0.4f}".format(mean_final_reward[episode]))
            self.logger.info("return standard deviation: {:0.4f}".format(std_final_reward[episode]))
            self.logger.info("min return: {:0.4f}; max return: {:0.4f}\n".format(min_final_reward[episode], max_final_reward[episode]))

            if self.plot_freq > 0 and episode % self.plot_freq == 0:
                self.plot(axs, episode, total_final_reward, losses, mean_avg_slowdowns, mean_avg_completion_times)

    def save(self, path, env):
        """
        Saves the model parameters into the specified file.
        """
        trained_params = self.get_params(self.opt_state)
        file = open(path, 'wb')
        pickle.dump(trained_params, file, -1)
        file.close()

    def load(self, path, env):
        """
        Loads the model parameters from the specified file.
        """
        self.logger.info("Loading model from %s" % (path))
        file = open(path, 'rb')
        params = pickle.load(file)

        # self.initialize_params(saved_params, self.input_shape)
        self.opt_state = self.opt_init(params)

    def plot(self, axs, episode, total_rewards, losses, avg_slowdowns, avg_completion_times):
        episode_seq = np.arange(episode+1)
        axs[0,0].clear()
        axs[0,0].plot(episode_seq, total_rewards[:episode+1], '-o')
        axs[0,0].set_title('Total reward')
        axs[0,0].set(xlabel='Episode', ylabel='Total reward')

        axs[0,1].plot(episode_seq, losses[:episode+1], '-o', color='green')
        axs[0,1].set_title('Loss')
        axs[0,1].set(xlabel='Episode', ylabel='Pseudo loss')

        axs[1,0].plot(episode_seq, avg_slowdowns[:episode+1], '-o', color='orange')
        axs[1,0].set_title('Average slowdown')
        axs[1,0].set(xlabel='Episode', ylabel='Average slowdown')

        axs[1,1].plot(episode_seq, avg_completion_times[:episode+1], '-o', color='red')
        axs[1,1].set_title('Average completion time')
        axs[1,1].set(xlabel='Episode', ylabel='Average completion time')

        plt.pause(0.05)

