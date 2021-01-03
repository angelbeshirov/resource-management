import numpy as np
from job import Job

class Machine:
    def __init__(self, number_resources, resource_slots, time_horizon):
        self.number_resources = number_resources
        self.resource_slots = resource_slots     # number of slots per resource

        self.running_jobs = [] # TODO: find the best structure for our problem (maybe list?)
        self.time_horizon = time_horizon
        self.available_slots = np.full((time_horizon, number_resources), resource_slots)

        # graphical representation
        self.colormap = np.arange(1 / float(40), 1, 1 / float(40))
        np.random.shuffle(self.colormap)
        self.canvas = np.zeros((self.number_resources, self.time_horizon, self.resource_slots))

    def reset(self):
        self.running_jobs = []
        self.canvas = np.zeros((self.number_resources, self.time_horizon, self.resource_slots))

    def allocate_job(self, job, current_time):
        # TODO: can job length be greater than time_horizon ?
        for t in range(self.time_horizon - job.length):
            # observe resources if job is executed
            new_available_slots = \
                self.available_slots[t:t+job.length, :] - job.resource_vector
            
            if np.all(new_available_slots >= 0):
                # update slots
                self.available_slots[t:t+job.length, :] = new_available_slots
                # set job start and end time
                job.set_start_time(current_time + t)
                # append job to runnign jobs
                self.running_jobs.append(job)
                # update graphics
                self.update_canvas(t, t + job.length, job)
                return True
        return False

    def update_canvas(self, start_time, end_time, job):
        color = self.get_unused_color()
        for r in range(self.number_resources):
            for i in range(start_time, end_time):
                available_slot = np.where(self.canvas[r, i, :] == 0)[0]
                self.canvas[r, i, available_slot[:job.resource_vector[r]]] = color

    def time_proceed(self, current_time):
        # move slots 1 position back (one time step passed)
        self.available_slots[:-1, :] = self.available_slots[1:, :]
        # set last resource as fully available (all its slots are free)
        self.available_slots[-1, :] = self.resource_slots

        # remove finished jobs from ruuning jobs list
        for job in self.running_jobs:
            if job.finish_time <= current_time:
                self.running_jobs.remove(job)
        
        # update canvas
        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0

    def calc_delay_panalty(self, delay_panalty):
        reward = 0
        for job in self.running_jobs:
            reward += delay_panalty / float(job.length)
        return reward

    def get_unused_color(self):
        used_colors = np.unique(self.canvas)
        for color in self.colormap:
            if color not in used_colors:
                return color
        return 0
    