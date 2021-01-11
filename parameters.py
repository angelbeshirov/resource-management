import numpy as np
import math

class Parameters:
    def __init__(self):
        self.number_resources = 2
        self.max_resource_slots = 25    # max number of available resource slots per resource
        self.episode_max_length = 200   # maximum number of time steps in an episode

        self.t = 4                      # duration parameter for the jobs
        self.r = 50                     # maximum resource request for new jobs
        self.jobs_sequence_length = 300  # length of one sequence of jobs, this parameter control the environemnt load
        self.simulation_length = 100    # number of job sequences
        self.job_rate = 0.75            # probability for a new job to arrive
        self.time_horizon = 15 * self.t # number of observed time steps
        self.work_queue_size = 10       # maximum number of waiting jobs in the queue, 10
        self.backlog_size = 60          # size of backlog queue

        self.delay_penalty = -1         # penalty for holding things in the current work screen
        self.hold_penalty = -1          # penalty for holding jobs in the queue
        self.dismiss_penalty = -1       # penalty for missing a job b/c of full queue

        # RMSProp
        self.learning_rate = 0.001      # the learning rate for training
        self.gamma = 0.9                # the gamma parameter for RMSProp
        self.eps = 1e-8                 # the eps parameter for RMSProp

        # self.discount = 1
        self.number_episodes = 1000    # number of episodes
        self.batch_size = 20           # the batch size

        # Network parameters
        self.backlog_width = int(math.ceil(self.backlog_size \
            / float(self.time_horizon)))                # parameter used for converting the backlog into (time_horizon, x) dimension

        self.input_height = self.time_horizon           # the network input height
        self.input_width = self.network_input_width = \
            int(0.5 * self.r +
             self.work_queue_size * 0.5 * self.r ) * self.number_resources + self.backlog_width # the network input width

        self.network_output_dim = self.work_queue_size + 1 # the output dimension from the policy (work_queue_size + 1, 1)
