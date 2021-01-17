import numpy as np
from job import Job

class Machine:
    """
    Class representing the cluster/machine which executes the jobs.
    """
    def __init__(self, number_resources, resource_slots, time_horizon, logger):
        self.number_resources = number_resources # number of resources in the machine
        self.resource_slots = resource_slots     # number of slots per resource
        self.time_horizon = time_horizon         # number of observable time steps

        self.running_jobs = [] # TODO: find the best structure for our problem (maybe list?)
        self.available_slots = np.full((time_horizon, number_resources), resource_slots)

        # graphical representation
        self.colormap = np.arange(1 / float(40), 1, 1 / float(40))
        self.canvas = np.zeros((self.number_resources, self.time_horizon, self.resource_slots))
        self.logger = logger

        np.random.shuffle(self.colormap)

    def reset(self):
        """
        Resets the machine to its initial state.
        """
        self.running_jobs = []
        self.canvas = np.zeros((self.number_resources, self.time_horizon, self.resource_slots))
        self.available_slots = np.full((self.time_horizon, self.number_resources), self.resource_slots)

    def allocate_job(self, job, current_time):
        """
        Tries to allocate a job if in the machine. The job must be with length not greater than the 
        observable time steps in the machine. If there are not enough resources currently in the machine 
        i.e. the job was not successfully allocated False will be returned, otherwise True.
        Future improvement can be to add job fragmentation.
        """
        for t in range(self.time_horizon - job.length):
            # observe resources if job is executed
            new_available_slots = \
                self.available_slots[t:t+job.length, :] - job.resource_vector
            
            if np.all(new_available_slots >= 0):
                # update slots
                self.available_slots[t:t+job.length, :] = new_available_slots
                # set job start and end time
                job.set_start_time(current_time + t)
                # append job to running jobs
                self.running_jobs.append(job)
                # update graphics
                self.update_canvas(t, t + job.length, job)
                return True
        return False

    def update_canvas(self, start_time, end_time, job):
        """
        Updates the canvas for rendering the machine state.
        """
        color = self.get_unused_color()
        for r in range(self.number_resources):
            for i in range(start_time, end_time):
                available_slot = np.where(self.canvas[r, i, :] == 0)[0]
                self.canvas[r, i, available_slot[:job.resource_vector[r]]] = color

    def time_proceed(self, current_time):
        """
        Moves the machine 1 time step further and updates the variables.
        """
        # move slots 1 position back (one time step passed)
        self.available_slots[:-1, :] = self.available_slots[1:, :]
        # set last resource as fully available (all its slots are free)
        self.available_slots[-1, :] = self.resource_slots

        # remove finished jobs from running jobs list
        for job in self.running_jobs:
            if job.finish_time <= current_time:
                self.running_jobs.remove(job)
                self.logger.debug("Job %s finished at %d" % (job.to_string(), current_time))
        
        # update canvas
        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0

    def calc_delay_panalty(self, delay_panalty):
        """
        The penalty is the sum of all -1/T_j where T_j is the length of 
        job currently in the system.
        """
        reward = 0
        for job in self.running_jobs:
            reward += delay_panalty / float(job.length)
        return reward

    def get_unused_color(self):
        """
        Returns the first unused color.
        """
        used_colors = np.unique(self.canvas)
        for color in self.colormap:
            if color not in used_colors:
                return color
        return 0

    def get_load(self):
        """
        Returns the load of the machine, at the current time step
        calculated by the formula:
        machine_load = (res_1_load + res_2_load + ... + res_n_load) / n,
        where n = number_resources and
        res_i_load = available_res_i_slots / (time_horizon * resource_slots)
        """
        res_loads = [np.sum(self.available_slots[:, res]) / (self.time_horizon * self.resource_slots) \
            for res in range(self.number_resources)]
        return np.mean(res_loads)
