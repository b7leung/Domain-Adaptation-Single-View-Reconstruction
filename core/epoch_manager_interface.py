from abc import ABC, abstractmethod


class EpochManager(ABC):
    """
    Abstract base class for an EpochManager
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_epoch(self, **kwargs):
        pass

    @abstractmethod
    def perform_step(self, **kwargs):
        pass

    @abstractmethod
    def end_epoch(self, **kwargs):
        pass
