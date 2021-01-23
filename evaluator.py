import numpy as np
import argparse
import matplotlib.pyplot as plt

from environment import ResourceManagementEnv, TerminationType
from parameters import Parameters
from data_generator import DataGenerator
from neural_network import Neural_network
from job import Job
from logger import LogLevel, Logger
from trainer import Trainer
import util

class Evaluator:
    def __init__(self, parameters, env, logger = Logger(LogLevel['info'])):
        self.env = env
        self.logger = logger
        self.episode_max_length = parameters.episode_max_length

    def evaluate_dnn(self, agent, deterministic=False):
        """
        Evaluates the Deep Neural Network agent for the particular environment. 
        """
        self.env.seq_number = 0
        self.env.reset()
        episode_length = self.episode_max_length

        self.logger.info("Evaluation of DNN agent started...")

        state = self.env.retrieve_state()
        for time_step in range(self.episode_max_length):
            pi_s = np.exp(agent.predict(state))
            
            action = np.argmax(pi_s) if deterministic else np.random.choice(self.env.actions, p = pi_s)
            # take an environment step
            _, _, done, allocation = self.env.step(action)
            
            while allocation == True: # freeze time
                state = self.env.retrieve_state()
                pi_s = np.exp(agent.predict(state))
                action = np.argmax(pi_s) if deterministic else np.random.choice(self.env.actions, p = pi_s)
                _ , _, done, allocation = self.env.step(action)

            if done:
                self.logger.info("No more jobs in the environment, everything is executed.")
                episode_length = time_step
                break
        
        slowdown = self.env.get_average_slowdown()
        self.logger.info("Evaluation completed in {} timesteps.".format(episode_length))
        self.logger.info("Final reward is {:.4f}.".format(self.env.reward()))
        self.logger.info("Average slowdown is {:.4f}".format(slowdown))

        return slowdown

    def evaluate_packer(self, agent):
        self.env.reset()
        episode_length = self.episode_max_length

        self.logger.info("Evaluation of packer agent started...")

        for time_step in range(self.episode_max_length):          
            action = agent.predict(self.env.job_queue)
            # take an environment step
            _ , _, done, allocation = self.env.step(action)

            while allocation == True: # freeze time
                action = agent.predict(self.env.job_queue)
                _ , _, done, allocation = self.env.step(action)

            if done:
                self.logger.info("No more jobs in the environment, everything is executed.")
                episode_length = time_step
                break
        
        slowdown = self.env.get_average_slowdown()
        self.logger.info("Evaluation completed in {} timesteps.".format(episode_length))
        self.logger.info("Final reward is {:.4f}.".format(self.env.reward()))
        self.logger.info("Average slowdown is {:.4f}".format(slowdown))

        return slowdown

    def evaluate_sjf(self, agent):
        self.env.reset()
        episode_length = self.episode_max_length

        self.logger.info("Evaluation of SJF agent started...")

        for time_step in range(self.episode_max_length):          
            action = agent.predict(self.env.job_queue)

            # take an environment step
            _ , _, done, allocation = self.env.step(action)
            while allocation == True: # freeze time
                action = agent.predict(self.env.job_queue)
                _ , _, done, allocation = self.env.step(action)

            if done:
                self.logger.info("No more jobs in the environment, everything is executed.")
                episode_length = time_step
                break
        
        slowdown = self.env.get_average_slowdown()
        self.logger.info("Evaluation completed in {} timesteps.".format(episode_length))
        self.logger.info("Final reward is {:.4f}.".format(self.env.reward()))
        self.logger.info("Average slowdown is {:.4f}".format(slowdown))

        return slowdown

