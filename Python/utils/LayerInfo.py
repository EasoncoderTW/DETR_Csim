from abc import ABC
from dataclasses import dataclass, asdict


class ShapeParam(ABC):
    def to_dict(self) -> dict:
        """Convert the object's attributes to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Create an instance of the class from a dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class Conv2DShapeParam(ShapeParam):
    """Follow the notation in the Eyeriss paper"""

    batch_size: int  # batch size
    input_height: int  # input height
    input_width: int  # input width
    filter_height: int  # filter height
    filter_width: int  # filter width
    output_height: int  # output height
    output_width: int  # output width
    input_channels: int  # input channels
    output_channels: int  # output channels
    stride: int = 1  # stride
    padding: int = 1  # padding

@dataclass(frozen=True)
class BatchNorm2DShapeParam(ShapeParam):
    batch_size: int  # batch size
    input_height: int  # input height
    input_width: int  # input width
    input_channels: int  # input channels

@dataclass(frozen=True)
class MultiHeadAttentionShapeParam(ShapeParam):
    batch_size: int  # batch size
    input_length: int  # input sequence length
    input_dim: int  # input dimension
    num_heads: int  # number of attention heads
    head_dim: int  # dimension of each attention head

@dataclass(frozen=True)
class LinearShapeParam(ShapeParam):
    batch_size: int  # batch size
    in_features: int
    out_features: int


@dataclass(frozen=True)
class MaxPool2DShapeParam(ShapeParam):
    batch_size: int  # batch size
    kernel_size: int
    stride: int

@dataclass(frozen=True)
class ActivationShapeParam(ShapeParam):
    batch_size: int  # batch size
    input_length: int  # input length
    input_dim: int  # input dimension

@dataclass(frozen=True)
class LayerNormShapeParam(ShapeParam):
    batch_size: int  # batch size
    input_length: int  # input length
    input_dim: int  # input dimension