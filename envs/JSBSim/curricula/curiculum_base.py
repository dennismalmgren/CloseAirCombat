import logging
from abc import ABC, abstractmethod


class BaseCurriculum(ABC):
    """
    Base Curriculum class
    Condition-specific get_termination and get_reward methods are implemented in subclasses
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod   
    def update_task(self, env):      
        raise NotImplementedError

    @abstractmethod   
    def reset(self, task, env):
        raise NotImplementedError
    
    def log(self, msg):
        logging.debug(msg)
