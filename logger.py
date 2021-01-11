

from enum import Enum

class LogLevel(Enum):
    debug = 1
    info = 2

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

class Logger:
    def __init__(self, log_level = LogLevel.info):
        self.log_level = log_level

    def debug(self, message):
        """
        Logs a debug message.
        """
        self.check_and_log(message, LogLevel.debug)

    def info(self, message):
        """
        Logs an info message.
        """
        self.check_and_log(message, LogLevel.info)

    def check_and_log(self, message, log_level):
        """
        Does the log level check and logs a message if the check passes.
        """
        if self.log_level <= log_level:
            print(message)