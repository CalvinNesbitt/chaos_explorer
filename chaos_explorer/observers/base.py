from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def look(self):
        "Look at the integrator"
        pass

    @abstractmethod
    def make_observations(self):
        "Make many observations"
        pass

    @abstractmethod
    def observations(self):
        "Return your observations"
        pass

    @abstractmethod
    def dump(self):
        "Dump your observations"
        pass

    @abstractmethod
    def wipe(self):
        "Wipe your observations"
        pass
