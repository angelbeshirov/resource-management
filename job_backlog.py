import numpy as np
from job import Job

# Class representing backlog queue
class JobBacklog:
    def __init__(self, size):
        self.size = size
        self.backlog = np.full(size, None)
        self.num_jobs = 0 # number of valid jobs in the backlog

    def empty(self):
        return self.backlog[0] == None
    
    # Removes job from backlog and returns it
    # Returns None if backlog is empty
    def dequeue(self):
        if self.empty():
            return None
        else:
            front_job = self.backlog[0]
            # shif remaining jobs
            self.backlog[:-1] = self.backlog[1:]
            self.backlog[-1] = None
            self.num_jobs -= 1
            return front_job

    def enqueue(self, job):
        if self.num_jobs < self.size:
            self.backlog[self.num_jobs] = job
            self.num_jobs += 1
            return True
        return False
    
    def calc_panalty(self, dismiss_penalty):
        penalty = 0
        for i in range(self.num_jobs):
            penalty += dismiss_penalty / float(self.backlog[i].length)