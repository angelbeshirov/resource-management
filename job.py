class Job:
    def __init__(self, resource_vector, length, id):
        self.resource_vector = resource_vector
        self.length = length
        self.id = id
        self.enter_time = -1
        self.start_time = -1
        self.finish_time = -1

    def set_start_time(self, start_time):
        self.start_time = start_time
        self.end_time = self.start_time + self.length

    def set_enter_time(self, time):
        self.enter_time = time