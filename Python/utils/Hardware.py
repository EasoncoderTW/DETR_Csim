from abc import ABC, abstractmethod
from .LayerInfo import *
from typing import Dict, Tuple, List, Any

class HardwareModel(ABC):
    """
    Abstract base class for hardware models.
    """
    def __init__(self, cycle_rate: float = 1e-9):
        self.cycle_rate = cycle_rate

    def latency(self, cycles: int) -> float:
        """
        Calculate the latency in seconds given the number of cycles.
        :param cycles: Number of cycles
        :return: Latency in seconds
        """
        return cycles * self.cycle_rate

    def __repr__(self):
        return f"{self.__class__.__name__}(cycle_rate={self.cycle_rate})"


class ComputeUnit(HardwareModel, ABC):
    """
    Represents a compute unit in the hardware model.
    """
    def __init__(self, cycle_rate: float = 1e-9):
        """
        Initialize the compute unit with cycle rate, number of units, and compute energy.
        :param cycle_rate: Cycle rate in seconds
        :param compute_energy: Energy consumed per compute operation
        :param suport_shape_param: Supported shape parameters for the compute unit
        """
        super().__init__(cycle_rate)
        self.support_shape_param = []

    @abstractmethod
    def latency(self, shape_param: ShapeParam) -> float:
        compute_cycles = 0
        return super().latency(compute_cycles)

    def __repr__(self):
        return f"{self.__class__.__name__}(cycle_rate={self.cycle_rate})"

class MemoryUnit(HardwareModel):
    """
    Represents a memory unit in the hardware model.
    """
    def __init__(self, cycle_rate: float = 1e-9, memory_bandwidth: float = 1e9):
        """
        Initialize the memory unit with cycle rate and memory bandwidth.
        :param cycle_rate: Cycle rate in seconds
        :param memory_bandwidth: Memory bandwidth in bytes per second
        """
        super().__init__(cycle_rate)
        self.memory_bandwidth = memory_bandwidth

    def latency(self, data_size: int) -> float:
        """
        Calculate the latency for a memory operation.
        :param data_size: Size of data in bytes
        :return: Latency in seconds
        """
        return data_size / self.memory_bandwidth * self.cycle_rate

    def __repr__(self):
        return f"{self.__class__.__name__}(cycle_rate={self.cycle_rate}, memory_bandwidth={self.memory_bandwidth})"


class Device:
    """
    Represents a hardware device with compute and memory units.
    """
    def __init__(self, name: str, compute_unit: Dict[str, ComputeUnit], memory_unit: Dict[str, MemoryUnit]):
        self.name = name
        self.compute_unit = compute_unit
        self.memory_unit = memory_unit

    def __repr__(self):
        s = f"[Device: {self.name}]\n"
        s += "\tCompute Units:\n"
        for name, unit in self.compute_unit.items():
            s += f"\t\t{name}: {unit}\n"
        s += "\tMemory Units:\n"
        for name, unit in self.memory_unit.items():
            s += f"\t\t{name}: {unit}\n"
        return s