import numpy as np
from job import Job

class JobBacklog:
    """
    Class representing backlog queue.
    """
    def __init__(self, size):
        self.size = size
        self.backlog = np.full(size, None)
        self.num_jobs = 0 # number of valid jobs in the backlog

    def empty(self):
        """
        Returns True if the backlog is empty and False otherwise.
        """
        return self.num_jobs == 0

    def front(self):
        """
        Returns the first element in the backlog.
        """
        return self.backlog[0]
    
    def dequeue(self):
        """
        Removes job from backlog and returns it. If the backlog is empty None is returned.
        """
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
        """
        Adds a job in the backlog if it is not already full. If the job was
        added successfully True is returned, otherwise this method returns False.
        """
        if self.num_jobs < self.size:
            self.backlog[self.num_jobs] = job
            self.num_jobs += 1
            return True
        return False
    
    def calc_panalty(self, dismiss_penalty):
        """
        The penalty here is the sum of all -1/T_j where T_j is the length of 
        job currently in the system.
        """
        penalty = 0
        for i in range(self.num_jobs):
            penalty += dismiss_penalty / float(self.backlog[i].length)
        return penalty