from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eqcycles.core.data import SimulationData

class BaseLoader(ABC):
    """
    Abstract Base Class for data loaders.
    Defines the interface for loading simulation data into a standardized format.
    """
    @abstractmethod
    def load(self, path: str, run_id: str) -> "SimulationData":
        """
        Loads simulation data from the specified path with a given run_id.

        Args:
            path: The base directory where simulation output files are located.
            run_id: The identifier suffix for the simulation files (e.g., '1', '205').

        Returns:
            A standardized SimulationData object.
        """
        pass
