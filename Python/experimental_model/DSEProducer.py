from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DSEProducerBASE(ABC):
    """
    Abstract base class for a DSE (Data Science Experiment) producer.
    This class defines the interface for producing DSEs.
    """
    def __init__(self):
        """
        Initialize the DSEProducer.
        This constructor can be extended by subclasses to initialize additional attributes.
        """
        self.produced_dses: List = []

    @abstractmethod
    def produce(self, dse: Dict[str, Any]) -> None:
        """
        Produce a DSE.

        :param dse: A dictionary representing the DSE to be produced.
        """
        pass

    @property
    def get_produced_dses(self) -> List[Dict[str, Any]]:
        """
        Get the list of produced DSEs.

        :return: A list of dictionaries representing the produced DSEs.
        """
        return self.produced_dses
