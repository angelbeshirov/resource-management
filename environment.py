import numpy as np
import matplotlib.pyplot as plt
import math

from enum import Enum
from parameters import Parameters
from machine import Machine
from data_generator import DataGenerator
from job import Job
from job_backlog import JobBacklog
# TODO: Add method/class description
class TerminationType(Enum):
    NoNewJob = 1    # terminate when sequence is empty
    AllJobsDone = 2 # terminate when sequence is empty and no jobs are running

class ResourceManagementEnv:
    def __init__(self, parameters, to_render = False, termination_type = TerminationType.NoNewJob, verbose = True):
        self.job_sequence_length = parameters.jobs_sequence_length
        self.simulation_length = parameters.simulation_length
        self.seq_number = 0 # index of current exemple sequence
        self.seq_idx = 0    # index in the current example sequence
        self.current_time = 0
        self.to_render = to_render
        self.verbose = verbose

        self.number_resources = parameters.number_resources
        self.input_height = parameters.input_height
        self.input_width = parameters.input_width
        self.max_resource_slots = parameters.max_resource_slots
        self.work_queue_size = parameters.work_queue_size
        self.backlog_size = parameters.backlog_size
        self.time_horizon = parameters.time_horizon

        self.data_generator = DataGenerator(parameters)
        self.job_queue = np.full(parameters.work_queue_size, None)

        # TODO: Might be good to initialize all Objects by passing the parameters object in the constructor
        # and then each object picks up whatever it needs from there
        self.job_backlog = JobBacklog(parameters.backlog_size) 
        self.termination_type = termination_type
        self.episode_max_length = parameters.episode_max_length
        self.machine = Machine(parameters.number_resources, \
            parameters.max_resource_slots,                  \
            parameters.time_horizon)
        self.work_sequences = self.generate_work_sequences() # Should this be generated each time?
        
        self.delay_penalty = parameters.delay_penalty
        self.hold_penalty = parameters.hold_penalty
        self.dismiss_penalty = parameters.dismiss_penalty
        self.actions = range(self.work_queue_size + 1) # +1 for the empty action
        
    def step(self, action):
        reward = 0
        done = False
        allocation = False

        # check if action is valid
        if action < len(self.job_queue) and self.job_queue[action] is not None:
            allocation = self.machine.allocate_job(self.job_queue[action], self.current_time)
        
        if allocation:
            self.log("Job (id: %d, length: %d, res_vec: %s) has started running" % \
                (self.job_queue[action].id, self.job_queue[action].length, np.array2string(self.job_queue[action].resource_vector)))
            # remove the job from the queue
            self.job_queue[action] = None
            # deque from job backlog
            if not self.job_backlog.empty():
                self.job_queue[action] = self.job_backlog.dequeue()
                self.log("Job (id: %d, length: %d, res_vec: %s) is added to queue (from backlog)" % \
                    (self.job_queue[action].id, self.job_queue[action].length, np.array2string(self.job_queue[action].resource_vector)))
        else:
            self.current_time += 1
            self.machine.time_proceed(self.current_time)

            # check whether to proceed
            done = self.done()
            
            if not done:
                # check if sequence is completed
                if self.seq_idx < self.job_sequence_length:
                    new_job = self.work_sequences[self.seq_number, self.seq_idx]
                    added_to_queue = False

                    if new_job is not None: # check if job is valid
                        new_job.set_enter_time(self.current_time)

                        for i in range(len(self.job_queue)):
                            if self.job_queue[i] is None:   # empty slot in the queue
                                self.job_queue[i] = new_job # put the new job in that slot
                                added_to_queue = True
                                self.log("Adding job (id: %d, length: %d, res_vec: %s) to queue" % \
                                    (new_job.id, new_job.length, np.array2string(new_job.resource_vector)))
                                break
                        if not added_to_queue:
                            self.log("Adding job (id: %d, length: %d, res_vec: %s) to backlog" % \
                                    (new_job.id, new_job.length, np.array2string(new_job.resource_vector)))
                            self.job_backlog.enqueue(new_job)
                    else: self.log("Skipping invalid job from job sequence.")
            # go to next job from sequence
            self.seq_idx += 1
            reward = self.reward()

        state = self.retrieve_state()
         
        if done:
            # we are done with current sequence
            self.seq_idx = 0
            self.seq_number = (self.seq_number + 1) % self.simulation_length
            self.reset()
        
        return state, reward, done

    
    def reward(self):
        reward = self.machine.calc_delay_panalty(self.delay_penalty)
        
        for job in self.job_queue:
            if job is not None:
                reward += self.hold_penalty / float(job.length)
        reward += self.job_backlog.calc_panalty(self.dismiss_penalty)
        return reward
        

    def done(self):
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
        self.seq_number = 0
        self.seq_idx = 0
        self.current_time = 0
        self.machine.reset()
        self.job_queue = np.full(self.work_queue_size, None)
        self.job_backlog = JobBacklog(self.backlog_size) 

    def render(self):
        # TODO: improve rendering (axis labels, titles, etc)
        rows = self.machine.number_resources
        cols = len(self.job_queue) + 1 + 1 # in one row display current resource, queue slots
        # TODO: add backlog display

        idx = 1

        for i in range(rows):
            plt.subplot(rows, cols, idx)
            idx += 1
            # display machine
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
                backlog_grid = np.zeros((self.job_backlog.size, 1))
                for b in range(self.job_backlog.num_jobs):
                    backlog_grid[b, 0] = 1
                plt.subplot(rows, cols, idx)
                plt.title('Backlog')
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

    # Current state shape is: 60x301
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
            state[:, column_iterator: column_iterator + self.max_resource_slots] = self.machine.canvas[i, :, :]
            column_iterator += self.max_resource_slots

            for j in range(self.work_queue_size):

                if self.job_queue[j] is not None:  # fill in a block of work if valid
                    state[: self.job_queue[j].length, \
                        column_iterator: column_iterator + self.job_queue[j].resource_vector[i]] = 1

                column_iterator += self.max_resource_slots

        state[: int(self.job_backlog.num_jobs / backlog_width), column_iterator: column_iterator + backlog_width] = 1
        if self.job_backlog.num_jobs % backlog_width > 0:
            state[self.job_backlog.num_jobs / backlog_width,
                       column_iterator: column_iterator + self.job_backlog.num_jobs % backlog_width] = 1
        column_iterator += backlog_width

        assert column_iterator == state.shape[1]

        return state

    def log(self, message):
        """
        Logs a message.
        """
        if self.verbose:
            print(message)