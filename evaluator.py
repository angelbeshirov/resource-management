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
        self.env.reset()
        total_reward = 0
        episode_length = self.episode_max_length

        self.logger.info("Evaluation of DNN agent started...")

        state = self.env.retrieve_state()
        for time_step in range(self.episode_max_length):
            pi_s = np.exp(agent.predict(state))
            
            action = np.argmax(pi_s) if deterministic else np.random.choice(self.env.actions, p = pi_s)

            # take an environment step
            _ , reward, done, allocation = self.env.step(action)
            while allocation == True:
                pi_s = np.exp(agent.predict(state))
                action = np.argmax(pi_s) if deterministic else np.random.choice(self.env.actions, p = pi_s)
                _ , reward, done, allocation = self.env.step(action)
            
            total_reward += reward

            # If everything is executed the rest of the rewards will be 0
            # which is exactly the expected behaviur since the environment
            # only returns negative rewards (-1/T_j)
            if done:
                self.logger.info("No more jobs in the environment, everything is executed.")
                episode_length = time_step
                break
        

        self.logger.info("Evaluation completed in {} timesteps.".format(episode_length))
        self.logger.info("Total reward is {}.".format(total_reward))
        self.logger.info("Average slowdown is {:.4f}".format(self.env.get_average_slowdown()))

    def evaluate_packer(self, agent):
        self.env.reset()
        total_reward = 0
        episode_length = self.episode_max_length

        self.logger.info("Evaluation of packer agent started...")

        for time_step in range(self.episode_max_length):          
            action = agent.predict(self.env.job_queue)
            #util.print_job_sequence(self.logger, self.env.job_queue)
            #self.logger.info("Action picked {}".format(action))
            # take an environment step
            _ , reward, done, allocation = self.env.step(action)

            while allocation == True:
                action = agent.predict(self.env.job_queue)
                #util.print_job_sequence(self.logger, self.env.job_queue)
                #self.logger.info("Action picked {}".format(action))
                _ , reward, done, allocation = self.env.step(action)
            
            total_reward += reward

            # If everything is executed the rest of the rewards will be 0
            # which is exactly the expected behaviur since the environment
            # only returns negative rewards (-1/T_j)
            if done:
                self.logger.info("No more jobs in the environment, everything is executed.")
                episode_length = time_step
                break
        

        self.logger.info("Evaluation completed in {} timesteps.".format(episode_length))
        self.logger.info("Total reward is {}.".format(total_reward))
        self.logger.info("Average slowdown is {:.4f}".format(self.env.get_average_slowdown()))

    def evaluate_sjf(self, agent):
        self.env.reset()
        total_reward = 0
        episode_length = self.episode_max_length
        #util.print_job_sequence(self.logger, self.env.job_queue)

        self.logger.info("Evaluation of SJF agent started...")

        for time_step in range(self.episode_max_length):          
            action = agent.predict(self.env.job_queue)
            #util.print_job_sequence(self.logger, self.env.job_queue)
            #self.logger.info("Action picked {}".format(action))

            # take an environment step
            _ , reward, done, allocation = self.env.step(action)
            
            while allocation == True:
                action = agent.predict(self.env.job_queue)
                #util.print_job_sequence(self.logger, self.env.job_queue)
                #self.logger.info("Action picked {}".format(action))
                _ , reward, done, allocation = self.env.step(action)
            
            total_reward += reward

            # If everything is executed the rest of the rewards will be 0
            # which is exactly the expected behaviur since the environment
            # only returns negative rewards (-1/T_j)
            if done:
                self.logger.info("No more jobs in the environment, everything is executed.")
                episode_length = time_step
                break
        

        self.logger.info("Evaluation completed in {} timesteps.".format(episode_length))
        self.logger.info("Total reward is {}.".format(total_reward))
        self.logger.info("Average slowdown is {:.4f}".format(self.env.get_average_slowdown()))

