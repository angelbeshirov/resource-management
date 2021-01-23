import numpy as np
from neural_network import Neural_network
from parameters import Parameters
from data_generator import DataGenerator
from environment import ResourceManagementEnv, TerminationType
from logger import Logger, LogLevel
from sys import maxsize

import matplotlib.pyplot as plt

import time
import random
import jax.numpy as jnp
import jax
import util
import copy
import os

class Trainer:
    """
    Does the neural network training based on multiple job sequences.
    """

    def __init__(self, parameters, logger = Logger(LogLevel['info']), plot_freq = 1):
        self.parameters = parameters
        self.logger = logger
        self.number_episodes = parameters.number_episodes
        self.plot_freq = plot_freq

    def train_sequence(self, nn, parameters, data, seq_number, result):
        """
        Retrieves MC samples for a single job sequence.
        """
        env = nn.env
        env.work_sequences = data
        env.seq_number = seq_number

        states = np.zeros((parameters.batch_size, parameters.episode_max_length, parameters.input_height * parameters.input_width), dtype=np.float32)
        actions = np.zeros((parameters.batch_size, parameters.episode_max_length), dtype=np.int)
        returns = np.zeros((parameters.batch_size, parameters.episode_max_length), dtype=np.float32)

        avg_slowdowns = np.zeros(parameters.batch_size, dtype=np.float32)
        avg_completion_times = np.zeros(parameters.batch_size, dtype=np.float32)
        system_loads = np.zeros(parameters.batch_size, dtype=np.float32)

        # get current policy network params
        current_params = nn.get_params(nn.opt_state)

        for j in range(parameters.batch_size):
            self.logger.debug("Monte carlo simulation %d started for seq %d" % (j, seq_number))
            
            # reset environment to the initial state
            env.reset()
        
            # zero rewards array (auxiliary array to store the rewards, and help compute the returns)
            rewards = np.zeros((parameters.episode_max_length, ), dtype=np.float32)

            # loop over steps in an episode
            for time_step in range(parameters.episode_max_length):
                # select state
                state = env.retrieve_state()
                states[j, time_step, :] = state
                
                # select an action according to current policy
                pi_s = np.exp(nn.predict_jax(current_params, state))
                action = np.random.choice(env.actions, p = pi_s)
                actions[j, time_step] = action
                
                # take an environment step, time is frozen until the agent picks
                # an invalid action or the empty (void) action. The last reward is 
                # taken into consideration (no rewards for intermediate steps)
                _, reward, done, allocation = env.step(action)
                while allocation == True:
                    state = env.retrieve_state()
                    pi_s = np.exp(nn.predict_jax(current_params, state))
                    action = np.random.choice(env.actions, p = pi_s)
                    _ , reward, done, allocation = env.step(action)

                # store reward
                rewards[time_step] = reward

                # If everything is executed the rest of the rewards will be 0
                # which is exactly the expected behaviour since the environment
                # only returns negative rewards (-1/T_j)
                if done:
                    self.logger.info("No more jobs in the environment, everything is executed.")
                    break
                
            # compute reward-to-go 
            # e.g. rewards = 1 2 3 4
            # rewards[::-1] = 4 3 2 1
            # cumsum -> 4 7 9 10
            # [::-1] -> 10 9 7 4
            returns[j,:] = jnp.cumsum(rewards[::-1])[::-1]
            avg_slowdowns[j] = env.get_average_slowdown()
            avg_completion_times[j] = env.get_average_completion_time()
            system_loads[j] = env.get_load()

        result.append({"states": states,
                       "actions": actions,
                       "returns": returns,
                       "avg_slowdowns": avg_slowdowns,
                       "avg_completion_time": avg_completion_times,
                       "system_loads": system_loads})

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
        if not os.path.exists('./logs'):
            os.makedirs('./logs')

        # Some additional raw file logging
        f = open("logs/log" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "a")
        f.write("episode,loss,total_reward,mean_reward,return_standard_deviation,min_return,max_return,ajs,ajc\n")

        best_reward = -maxsize   # get the max value
        best_slowdown = maxsize  # get the min value
        env = ResourceManagementEnv(self.parameters, self.logger, to_render=False, termination_type=TerminationType.AllJobsDone)
        simulation_length = self.parameters.simulation_length
        data = env.generate_work_sequences()

        nn = Neural_network(self.parameters, env, self.logger)

        _, axs = plt.subplots(2, 3, figsize=(16,10))

        self.logger.info("Starting training")

        # mean reward at the end of the episode for all job sequences
        mean_final_reward = np.zeros(self.number_episodes, dtype=np.float32)
        # total reward at the end of the episode for all job sequences
        total_final_reward = np.zeros_like(mean_final_reward)
        # standard deviation of the reward at the end of the episode for all job sequences
        std_final_reward = np.zeros_like(mean_final_reward)
        # batch minimum at the end of the episode for all job sequences
        min_final_reward = np.zeros_like(mean_final_reward) 
        # batch maximum at the end of the episode for all job sequences
        max_final_reward = np.zeros_like(mean_final_reward)
        # average slowdown at the end of the episode (slowdown = Sum -1/T_j (T_j = duration of jobs in the system)) for all job sequences
        mean_avg_slowdowns = np.zeros_like(mean_final_reward)
        # average completion time at the end of the episode (completion_time = finish_time - enter_time) for all job sequences
        mean_avg_completion_times = np.zeros_like(mean_final_reward)
        # average system load at the end of episode for all job sequences
        avg_system_loads = np.zeros_like(mean_final_reward)

        losses = []
        for episode in range(self.number_episodes): 
            self.logger.info("Episode %d started" % episode)
            start_time = time.time()
            indices = list(range(simulation_length))
            np.random.shuffle(indices)
            result = []
            copy_data = copy.deepcopy(data)

            for s in range(simulation_length):
                self.train_sequence(nn, self.parameters, copy_data, indices[s], result)

            loss = nn.update(result)
            losses.append(loss)
            ### record time needed for a single epoch
            episode_time = time.time() - start_time
            returns = np.concatenate([r["returns"] for r in result])
            avg_slowdowns = np.concatenate([r["avg_slowdowns"] for r in result])
            avg_completion_times = np.concatenate([r["avg_completion_time"] for r in result])
            system_loads = np.concatenate([r["system_loads"] for r in result])
            # check performance
            mean_final_reward[episode]=jnp.mean(returns[:,-1])
            total_final_reward[episode] = jnp.sum(returns[:,-1])
            std_final_reward[episode] =jnp.std(returns[:,-1])
            min_final_reward[episode], max_final_reward[episode] = np.min(returns[:,-1]), np.max(returns[:,-1])
            mean_avg_slowdowns[episode] = jnp.mean(avg_slowdowns)
            mean_avg_completion_times[episode] = jnp.mean(avg_completion_times)
            avg_system_loads[episode] = jnp.mean(system_loads)

            if total_final_reward[episode] >= best_reward:
                self.logger.info("Saving new best reward model with reward %d in episode %d" % (total_final_reward[episode], episode))
                nn.save("./models/best_model_reward_" + time.strftime("%Y%m%d-%H%M%S") + ".pkl")
                best_reward = total_final_reward[episode]

            if mean_avg_slowdowns[episode] <= best_slowdown:
                self.logger.info("Saving new best slowdown model with slowdown %d in episode %d" % (mean_avg_slowdowns[episode], episode))
                nn.save("./models/best_model_slowdown_" + time.strftime("%Y%m%d-%H%M%S") + ".pkl")
                best_slowdown = mean_avg_slowdowns[episode]

            self.logger.info("episode {} in {:0.2f} sec".format(episode, episode_time))
            self.logger.info("loss: {:0.4f}".format(loss))
            self.logger.info("total reward: {:0.4f}".format(total_final_reward[episode]))
            self.logger.info("mean reward: {:0.4f}".format(mean_final_reward[episode]))
            self.logger.info("return standard deviation: {:0.4f}".format(std_final_reward[episode]))
            self.logger.info("min return: {:0.4f}; max return: {:0.4f}".format(min_final_reward[episode], max_final_reward[episode]))
            self.logger.info("average job slowdown: {:0.4f}".format(mean_avg_slowdowns[episode]))
            self.logger.info("average job completion time: {:0.4f}\n".format(mean_avg_completion_times[episode]))
            f.write("{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f}\n" \
                .format(episode,loss,total_final_reward[episode],mean_final_reward[episode],std_final_reward[episode],\
                    min_final_reward[episode],max_final_reward[episode],mean_avg_slowdowns[episode],mean_avg_completion_times[episode]))

            if self.plot_freq > 0 and episode % self.plot_freq == 0:
                episode_seq = np.arange(episode + 1)
                util.plot_total_rewards(axs[0,0], episode_seq, total_final_reward, best_reward)
                util.plot_losses(axs[0,1], episode_seq, losses)
                util.plot_min_max_rewards(axs[0,2], episode_seq, min_final_reward, max_final_reward, mean_final_reward, std_final_reward)
                util.plot_avg_slowdowns(axs[1,0], episode_seq, mean_avg_slowdowns)
                util.plot_avg_completion_time(axs[1,1], episode_seq, mean_avg_completion_times)
                util.plot_system_load(axs[1,2], episode_seq, avg_system_loads)
                plt.pause(0.05)
        f.close()