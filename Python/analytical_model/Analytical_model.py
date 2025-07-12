from typing import Dict, Any, Tuple, List, Type
from torch import nn
import torch
from utils.LayerInfo import *
from utils.ModelParser import ModelParser
from utils.Hardware import *

class AnalyticalModel:
    def __init__(self,
                 model: nn.Module,
                 input_shape: tuple = (1, 3, 800, 800),
                 device: List[Device] = None):
        self.model: nn.Module = model
        self.input_shape = input_shape  # Example input shape for image models
        self.parser = ModelParser(model, input_shape)
        self.device: List[Device] = device if device else []

    def parse_model(self) -> Dict[str, Any]:
        """
        Parses the model to extract its architecture and parameters.
        Returns a dictionary containing the model's architecture and parameters.
        """
        return self.parser.parse(verbose=False)

    def get_hardware(self, ShapeParamType: ShapeParam) -> Tuple[str, HardwareModel]:
        """
        Returns the hardware model and shape parameters for a given ShapeParam type.
        """
        for hardware in self.device:
            if isinstance(hardware, Device):
                for name, compute_unit in hardware.compute_unit.items():
                    if ShapeParamType in compute_unit.support_shape_param:
                        return name, compute_unit

        return None, None

    def analyze(self) -> Dict[str, Any]:
        """
        Analyzes the model and returns a dictionary with analysis results.
        This method should be implemented by subclasses.
        """
        print("Starting model analysis...")
        print("Device information:")
        for dev in self.device:
            print(dev)

        latency_total = 0
        for module, param in self.parser.shape_params.items():
            name, hw_model = self.get_hardware(type(param))
            if hw_model:
                print(f"Using hardware model: {name}({hw_model}) for {module.__class__.__name__}", end=' ')
                # Perform analysis using the hardware model
                latency = hw_model.latency(param)
                print(f"Latency: {latency}")
            else:
                print(f"No hardware model found for {module.__class__.__name__}")
            latency_total += latency if hw_model else 0

        print(f"Total latency: {latency_total} seconds")
        print(f"FPS: {1 / latency_total if latency_total > 0 else float('inf')}")


    def visualize(self) -> None:
        """
        Visualizes the model's architecture or analysis results.
        This method should be implemented by subclasses.
        """
        pass
