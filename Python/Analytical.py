import torch
from analytical_model import *


class GeneralComputeUnit(ComputeUnit):
    def __init__(self, cycle_rate: float):
        super().__init__(cycle_rate)
        self.support_shape_param = [Conv2DShapeParam, LinearShapeParam, MultiHeadAttentionShapeParam, ActivationShapeParam]

    def latency(self, shape_param: ShapeParam) -> float:
        # Example implementation for general compute unit latency calculation
        if isinstance(shape_param, Conv2DShapeParam):
            compute_cycles = shape_param.input_height * shape_param.input_width * shape_param.input_channels * shape_param.filter_height * shape_param.filter_width * shape_param.output_channels / 1024
        elif isinstance(shape_param, LinearShapeParam):
            compute_cycles = shape_param.in_features * shape_param.out_features / 1024
        else:
            compute_cycles = 0
        return compute_cycles * self.cycle_rate


def main():
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    analytical_model = AnalyticalModel(
        model=model,
        input_shape=(1, 3, 800, 800),
        device=[
            Device(name='GeneralComputeUnit', compute_unit={
                'general': GeneralComputeUnit(cycle_rate=1e-9)
            }, memory_unit={
                'general_memory': MemoryUnit(cycle_rate=1e-9, memory_bandwidth=1e9)
            } )
        ]
    )

    analytical_model.parse_model()
    analytical_model.analyze()

if __name__ == "__main__":
    main()