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

class PackerAgent:
    def __init__(self, parameters, env, logger = Logger(LogLevel['info'])):
        self.work_queue_size = parameters.work_queue_size
        self.time_horizon = parameters.time_horizon
        self.env = env
        self.logger = logger

    def predict(self, work_queue):
        """
        Packer agent picks the (task,machine) pair with the highest dot product.
        Job = (resource_requirements, job_len)
        MachineAvailableSlots = (time_horizon, resoruce)
        If the job can be allocated at avbl_res = (x:x+job_len) then evaluate 
        avbl_res[0,:] * job_resource_vec and find the max among the work queue.
        """
        best_score = 0
        action = self.work_queue_size # the action to skip allocation

        for i in range(len(work_queue)):
            job = work_queue[i]
            if job is None:
                continue

            for t in range(self.time_horizon - job.length):
                available_slots = \
                    self.env.machine.available_slots[t:t+job.length, :] - job.resource_vector # job_len,2

                if np.all(available_slots >= 0):
                    tmp_score = jnp.dot(available_slots[0].flatten(), job.resource_vector)
                    
                    if tmp_score > best_score:
                        best_score = tmp_score
                        action = i
        return action


class SJFAgent:
    def __init__(self, parameters, env, logger = Logger(LogLevel['info'])):
        self.work_queue_size = parameters.work_queue_size
        self.time_horizon = parameters.time_horizon
        self.env = env
        self.logger = logger

    def predict(self, work_queue):
        """
        Shortest job first (SJF) agent picks the job with highest SJF score
        SJF score = 1 / T, where T is the work length
        """
        action = self.work_queue_size
        max_sjf_score = 0

        for i in range(self.work_queue_size):
            job = work_queue[i]
            if job is not None:
                for t in range(self.time_horizon - job.length):
                    available_slots = \
                        self.env.machine.available_slots[t:t+job.length, :] - job.resource_vector
                    
                    if np.all(available_slots >= 0):
                        tmp_score = 1 / float(job.length)

                        if tmp_score > max_sjf_score:
                            max_sjf_score = tmp_score
                            action = i
        return action