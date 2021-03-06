import numpy as np
import matplotlib.pyplot as plt
import math

from enum import Enum
from parameters import Parameters
from machine import Machine
from data_generator import DataGenerator
from job import Job
from job_backlog import JobBacklog
from logger import Logger, LogLevel

class TerminationType(Enum):
    NoNewJob = 1    # terminate when sequence is empty
    AllJobsDone = 2 # terminate when sequence is empty and no jobs are running

class ResourceManagementEnv:
    """
    Represents the environment containing all the jobs and performing 
    the actions(job scheduling) through which the agent interacts with it.
    """
    def __init__(self, parameters, logger = Logger(LogLevel['info']), to_render = False, termination_type = TerminationType.NoNewJob):
        self.seq_number = 0         # index of current exemple sequence
        self.seq_idx = 0            # index in the current example sequence
        self.current_time = 0       # current system time
        self.current_queue_size = 0 # current queue size

        self.to_render = to_render  # for visualization
        self.termination_type = termination_type # the termination type

        self.job_sequence_length = parameters.jobs_sequence_length  # sequence of the job length
        self.simulation_length = parameters.simulation_length       # number of sequences
        self.number_resources = parameters.number_resources         # number of resources
        self.input_height = parameters.input_height                 # input height
        self.input_width = parameters.input_width                   # input width
        self.max_resource_slots = parameters.max_resource_slots     # max resources slots of each resource
        self.work_queue_size = parameters.work_queue_size           # size of the work queue
        self.backlog_size = parameters.backlog_size                 # size of the backlog
        self.time_horizon = parameters.time_horizon                 # time horizon (fixed time in the machine)

        self.dismiss_penalty = parameters.dismiss_penalty           # dismiss penalty, used if we want to penalize the backlog jobs
        self.hold_penalty = parameters.hold_penalty                 # hold penalty, used if we want to penalize the jobs in the queue
        self.delay_penalty = parameters.delay_penalty               # delay penalty, used to penalize the jobs in the machine
        self.episode_max_length = parameters.episode_max_length     # length of the episode

        self.data_generator = DataGenerator(parameters)             # data generator for new sequences
        self.job_queue = np.full(self.work_queue_size, None)        # job queue
        self.logger = logger                                        # logger    

        self.job_backlog = JobBacklog(parameters.backlog_size)      # size of the backlog
        self.machine = Machine(parameters.number_resources, \
            parameters.max_resource_slots,                  \
            parameters.time_horizon,
            logger)                                                 # the machine

        self.work_sequences = self.generate_work_sequences()        # work sequences
        self.actions = range(self.work_queue_size + 1)              # actions which can be taken, 0-n-1 schedule job at that indes +1 for the empty action

    def step(self, action):
        """
        Performs a step in the environment. If the allocation was successful and the backlog is not empty
        moves the first backlog job into the working queue. If the allocation was not successful the next
        job from the sequence will either be added to the working queue if there is available slot
        or to the backlog otherwise.
        Returns state, reward, done triplet.
        """
        reward = 0
        done = False
        allocation = False

        # check if action is valid
        if action < len(self.job_queue) and self.job_queue[action] is not None:
            allocation = self.machine.allocate_job(self.job_queue[action], self.current_time)

        if allocation:
            self.logger.debug("Job (id: %d, length: %d, res_vec: %s) has started running at %d" % \
                (self.job_queue[action].id, self.job_queue[action].length, np.array2string(self.job_queue[action].resource_vector), self.current_time))
            # remove the job from the queue
            self.job_queue[action] = None
            self.current_queue_size -= 1
            # deque from job backlog
            if not self.job_backlog.empty():
                self.job_queue[action] = self.job_backlog.dequeue()
                self.logger.debug("Job (id: %d, length: %d, res_vec: %s) is added to queue (from backlog)" % \
                    (self.job_queue[action].id, self.job_queue[action].length, np.array2string(self.job_queue[action].resource_vector)))
                self.current_queue_size += 1
        else:
            self.current_time += 1
            self.machine.time_proceed(self.current_time)

            # check whether to proceed
            done = self.done()
            reward = self.reward()
            
            if not done:
                self.fill_up_queue_and_backlog()
        
        state = self.retrieve_state()
        
        # we also need the allocation, because of the artificial pause of the time (allowing multiple actions to be taken at each timestep)
        return state, reward, done, allocation 

    def next_jobset(self):
        """
        Go to the next job sequence.
        """
        self.seq_number = (self.seq_number + 1) % self.simulation_length

    def fill_up_queue_and_backlog(self):
        """
        Fill up the queue and backlog.
        """
        if self.done():
            return
        # Fill up the queue from the backlog
        while not self.job_backlog.empty() and self.current_queue_size < self.work_queue_size:
            backlog_job = self.job_backlog.dequeue()
            for i in range(len(self.job_queue)):
                if self.job_queue[i] is None:
                    self.job_queue[i] = backlog_job
                    self.current_queue_size += 1
                    break
        
        # Fill up the queue from the current simulation (if the backlog was not enough or empty)
        while self.seq_idx < self.job_sequence_length and self.current_queue_size < self.work_queue_size:
            new_job = self.work_sequences[self.seq_number, self.seq_idx]
            if new_job is not None: # check if job is valid
                new_job.set_enter_time(self.current_time)
                for i in range(len(self.job_queue)):
                    if self.job_queue[i] is None:
                        self.job_queue[i] = new_job
                        self.current_queue_size += 1
                        break
            self.seq_idx += 1

        # Fill up the backlog from the simulation
        while self.seq_idx < self.job_sequence_length:
            new_job = self.work_sequences[self.seq_number, self.seq_idx]
            if new_job is None: # check if job is valid
                self.seq_idx += 1
                continue
            else:
                new_job.set_enter_time(self.current_time)
                if self.job_backlog.enqueue(new_job) == True:
                    self.seq_idx += 1
                else:
                    break
    
    def print_work_sequence(self):
        """
        Print the current work sequence.
        """
        for x in self.work_sequences:
            for job in x:
                if job is not None:
                    print(job.to_string())
                else:
                    print("Job is None")
    
    def reward(self):
        """
        Calculates the reward of the environment. The reward is the sum of -1/T_j where
        T_j is the length of each job in the system either currently scheduled, 
        in the job_queue waiting to be scheduled or in the backlog.
        """
        reward = self.machine.calc_delay_panalty(self.delay_penalty)
        
        for job in self.job_queue:
            if job is not None:
                reward += self.hold_penalty / float(job.length)
        reward += self.job_backlog.calc_panalty(self.dismiss_penalty) # uncomment to add the reward from the backlog
        return reward


    def reward_completion(self):
        """
        Reward to optimize for completion time. -|J|, where J is the 
        jobs currently in the system.
        """
        reward = len(self.machine.running_jobs)
        for job in self.job_queue:
            if job is not None:
                reward += 1
        reward += self.job_backlog.num_jobs # uncomment to add the reward from the backlog

        return -reward

    def done(self):
        """
        Checks if the environment has anymore valid actions to be 
        performed based on the termination type. For training the 
        termination type should be AllJobsDone i.e. all jobs have 
        finished executing or the time_step exceeds the episode length.
        """
        if self.termination_type == TerminationType.NoNewJob:
            return self.seq_idx >= self.job_sequence_length # current sequence is completed
        elif self.termination_type == TerminationType.AllJobsDone:
            if self.seq_idx >= self.job_sequence_length and      \
                len(self.machine.running_jobs) == 0 and          \
                all(slot is None for slot in self.job_queue) and \
                self.job_backlog.empty():
                return True
            elif self.current_time > self.episode_max_length:
                return True
        return False
        

    def reset(self):
        """
        Resets the environment into it's initial state.
        """
        self.current_queue_size = 0
        self.seq_idx = 0
        self.current_time = 0
        self.machine.reset()
        self.job_queue = np.full(self.work_queue_size, None)
        self.job_backlog = JobBacklog(self.backlog_size) 

        # The queue and backlog should be filled up beforehand
        self.fill_up_queue_and_backlog()

    def render(self):
        """
        Renders the state of the environment. Representing the cluster, the queue and the backlog.
        """
        rows = self.machine.number_resources
        cols = len(self.job_queue) + 1 + 1 # display machine, the queue and the backlog

        idx = 1

        for i in range(rows):
            plt.subplot(rows, cols, idx)
            idx += 1
            # display machine
            plt.title('Cluster')
            plt.xlabel('resource slots')
            plt.ylabel('time')
            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)
            for j in range(len(self.job_queue)):
                slot_grid = np.zeros((self.machine.time_horizon, self.machine.resource_slots))
                if self.job_queue[j] is not None:
                    job = self.job_queue[j]
                    slot_grid[:job.length, :job.resource_vector[i]] = 1
                
                plt.subplot(rows, cols, idx)
                plt.title('Job slot %d' % (j+1))
                idx += 1
                plt.imshow(slot_grid, interpolation='nearest', vmax=1)
            if i == 0: # show backlog only on the first line
                backlog_grid = np.zeros((self.job_backlog.size, 5))
                for b in range(self.job_backlog.num_jobs):
                    backlog_grid[b, :] = 1
                plt.subplot(rows, cols, idx)
                plt.title('Backlog')
                plt.xticks([])
                plt.imshow(backlog_grid, interpolation='nearest', vmax=1)
            idx += 1
        plt.show()

    def generate_work_sequences(self):
        """
        Generates work sequences
        Output: @simulation_length x @job_sequence_length array of type Job
        """
        counter = 1
        work_sequences = np.full((self.simulation_length, self.job_sequence_length), None, dtype=object)
        for i in range(self.simulation_length):
            job_lengths, job_resource_vectors = self.data_generator.generate_sequence()
            for j in range(self.job_sequence_length):
                # check if valid Job
                if job_lengths[j] > 0:
                    work_sequences[i, j] = Job(job_resource_vectors[j], \
                        job_lengths[j],                                 \
                        id=counter)
                    counter = counter + 1
        return work_sequences

    def retrieve_state(self):
        """
        Construct the state of the environment, used as input to the neural network policy during training.
        Note: The backlog representation should be in a format suitable for the neural network, which means
        to be of shape time_horizonXsomething. To achieve this we divide the backlog size to the time_horizon.
        This keeps the first dimension fixed regardless of the backlog size.
        """
        backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon))) 
        state = np.zeros((self.input_height, self.input_width))

        column_iterator = 0

        for i in range(self.number_resources):
            state[:, column_iterator] = self.machine.resource_slots - self.machine.available_slots[:, i] # 2x60x25
            column_iterator += 1

            for j in range(self.work_queue_size):

                if self.job_queue[j] is not None:  # fill in a block of work if valid
                    state[: self.job_queue[j].length, \
                        column_iterator] = self.job_queue[j].resource_vector[i]
                column_iterator += 1

        if self.job_backlog.num_jobs <= self.input_height: # observe the backlog
            state[: self.job_backlog.num_jobs, column_iterator] = 1
            column_iterator += backlog_width
        else:
            aux = int(math.floor(self.job_backlog.num_jobs / float(self.time_horizon)))
            state[:, column_iterator: (column_iterator + aux)] = 1
            column_iterator += aux
            state[: (int(self.job_backlog.num_jobs % self.input_height)), column_iterator] = 1
            column_iterator += 1
        assert column_iterator == state.shape[1]

        return state.flatten()

    def retrieve_compact_state(self):
        """
        Compact form of the state. Not very useful, but still can be experimented with further.
        """
        state = np.zeros(self.time_horizon * (self.number_resources + 1) + self.work_queue_size * (self.number_resources + 1) + 1)

        iterator = 0

        # current work reward, after each time step, how many jobs left in the machine
        job_allocated = np.ones(self.time_horizon) * len(self.machine.running_jobs)
        for j in self.machine.running_jobs:
            job_allocated[j.finish_time - self.current_time: ] -= 1

        state[iterator: iterator + self.time_horizon] = job_allocated
        iterator += self.time_horizon

        # current work available slots
        for i in range(self.number_resources):
            state[iterator: iterator + self.time_horizon] = self.machine.available_slots[:, i]
            iterator += self.time_horizon

        # new work duration and size
        for i in range(self.work_queue_size):

            if self.job_queue[i] is None:
                state[iterator: iterator + self.number_resources + 1] = 0
                iterator += self.number_resources + 1
            else:
                state[iterator] = self.job_queue[i].length
                iterator += 1

                for j in range(self.number_resources):
                    state[iterator] = self.job_queue[i].resource_vector[j]
                    iterator += 1

        # backlog queue
        state[iterator] = self.job_backlog.num_jobs
        iterator += 1

        assert iterator == len(state)  # fill up the compact representation vector

        return state

    def get_average_slowdown(self):
        """
        Calculates average slowdown over all finished jobs
        """
        slowdowns_sum = 0
        finished_jobs_cnt = 0
        for seq in range(len(self.work_sequences)):
            for job in self.work_sequences[seq]:
                if job is not None and job.finish_time > -1 \
                    and job.finish_time <= self.current_time:
                    slowdowns_sum += ((job.finish_time - job.enter_time) / float(job.length))
                    finished_jobs_cnt += 1

        if finished_jobs_cnt == 0: return 0
        return slowdowns_sum / float(finished_jobs_cnt)
    
    def get_average_completion_time(self):
        """
        Calculates the average job completion time of all finished jobs
        """
        completion_times_sum = 0
        finished_jobs_cnt = 0
        for seq in range(len(self.work_sequences)):
            for job in self.work_sequences[seq]:
                if job is not None and job.finish_time > -1 \
                    and job.finish_time <= self.current_time:
                    completion_times_sum += (job.finish_time - job.enter_time)
                    finished_jobs_cnt += 1

        if finished_jobs_cnt == 0: return 0
        return completion_times_sum / float(finished_jobs_cnt)

    def get_queue_load(self):
        """
        Returns queue load, calculated by the formula
        queue_load = jobs_in_queue / queue_size,
        where jobs_in_queue = the number on non-None
        slots in the queue
        """
        return np.mean([1 if job is not None else 0 for job in self.job_queue])

    def get_load(self):
        """
        Returns the overall load of the environement, that is the queue load + machine load.
        """
        return self.get_queue_load() + self.machine.get_load()