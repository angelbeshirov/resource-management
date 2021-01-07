import jax.numpy as jnp
import numpy as np
import time
import itertools

from jax import random
from jax.experimental import stax 
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax import grad
from jax.tree_util import tree_flatten

from jax.experimental import optimizers # gradient descent optimizers
from jax import jit

import numpy as np
from parameters import Parameters
from data_generator import DataGenerator
from environment import ResourceManagementEnv

class Neural_network:
    def __init__(self, parameters):
        self.current_time = 0
        self.seed = 0
        self.number_episodes = parameters.number_episodes
        self.batch_size = parameters.batch_size
        self.episode_max_length = parameters.episode_max_length

        self.input_height = parameters.input_height
        self.input_width = parameters.input_width

        self.learning_rate = parameters.learning_rate
        self.eps = parameters.eps
        self.gamma = parameters.gamma

        rng = random.PRNGKey(self.seed)

        self.initialize_params, self.predict_jax = stax.serial(
                                            Dense(20),
                                            Relu,
                                            Dense(parameters.network_output_dim),
                                            LogSoftmax
        )

        input_shape = (-1, self.episode_max_length, self.input_height * self.input_width)
        self.output_shape, self.inital_params = self.initialize_params(rng, input_shape)

        self.opt_init, self.opt_update, self.get_params = \
            optimizers.rmsprop(step_size = self.learning_rate, gamma = self.gamma, eps = self.eps)
            # optimizers.adam(step_size = self.learning_rate, b1 = self.b1, b2 = self.b2, eps = self.eps)

        self.opt_state = self.opt_init(self.inital_params)
        self.step = 0
        

        print('Output shape of the model is {}.\n'.format(self.output_shape))

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
        current_params = self.get_params(self.opt_state)
        grad_params = grad(self.pseudo_loss)(current_params, batch)
        self.opt_state = self.opt_update(self.step, grad_params, self.opt_state)
        self.step = self.step + 1

    def predict(self, data):
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
    def train(self, env):
        # preallocate data using arrays initialized with zeros
        state=np.zeros((self.input_height * self.input_width,), dtype=np.float32)
    
        states = np.zeros((self.batch_size, self.episode_max_length, self.input_height * self.input_width), dtype=np.float32)
        actions = np.zeros((self.batch_size, self.episode_max_length), dtype=np.int)
        returns = np.zeros((self.batch_size, self.episode_max_length), dtype=np.float32)
    
        # mean reward at the end of the episode
        mean_final_reward = np.zeros(self.number_episodes, dtype=np.float32)
        # standard deviation of the reward at the end of the episode
        std_final_reward = np.zeros_like(mean_final_reward)
        # batch minimum at the end of the episode
        min_final_reward = np.zeros_like(mean_final_reward)
        # batch maximum at the end of the episode
        max_final_reward = np.zeros_like(mean_final_reward)

        print("\nStarting training...\n")

        # set the initial model parameters in the optimizer
        self.opt_state = self.opt_init(self.inital_params)

        # loop over the number of training episodes
        for episode in range(self.number_episodes): 
            
            ### record time
            start_time = time.time()
            
            # get current policy  network params
            current_params = self.get_params(self.opt_state)
            
            # MC sample
            for j in range(self.batch_size):
                
                # reset environment to a random initial state
                #env.reset(random=False) # fixed initial state
                env.reset()
            
                # zero rewards array (auxiliary array to store the rewards, and help compute the returns)
                rewards = np.zeros((self.episode_max_length, ), dtype=np.float32)
            
                # loop over steps in an episode
                for time_step in range(self.episode_max_length):

                    # select state
                    state = env.retrieve_state().reshape(1, -1)
                    states[j, time_step, :] = state

                    # select an action according to current policy
                    pi_s = np.exp(self.predict_jax(current_params, state))
                    action = np.random.choice(env.actions, p = pi_s[0])
                    actions[j,time_step] = action

                    # take an environment step
                    _ , reward, _ = env.step(action)
                    # state[:] = s.reshape([1, -1])

                    # store reward
                    rewards[time_step] = reward
                    
                # compute reward-to-go 
                returns[j,:] = jnp.cumsum(rewards[::-1])[::-1]
                
                
                    
            # define batch of data
            trajectory_batch = (states, actions, returns)
            
            # update model
            self.update(trajectory_batch)
                    
            ### record time needed for a single epoch
            episode_time = time.time() - start_time
            
            # check performance
            mean_final_reward[episode]=jnp.mean(returns[:,-1])
            std_final_reward[episode] =jnp.std(returns[:,-1])
            min_final_reward[episode], max_final_reward[episode] = np.min(returns[:,-1]), np.max(returns[:,-1])

            
            # print results every 10 epochs
            #if episode % 5 == 0:
            print("episode {} in {:0.2f} sec".format(episode, episode_time))
            print("mean reward: {:0.4f}".format(mean_final_reward[episode]) )
            print("return standard deviation: {:0.4f}".format(std_final_reward[episode]) )
            print("min return: {:0.4f}; max return: {:0.4f}\n".format(min_final_reward[episode], max_final_reward[episode]) )
