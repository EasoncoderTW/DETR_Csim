from dataclasses import dataclass
import hashlib
import json
import json

# -----------------------------
# Parameter Definition
# -----------------------------
@dataclass
class SramTileParameters:
    CONV_TILED_OUT_HEIGHT: int = 32
    SRAM_DEFAULT_SIZE: int = 8388608 # 8 MB
    CONV_TILED_OUT_WIDTH: int = 32
    CONV_TILED_OUT_CHANNELS: int = 128
    CONV_TILED_IN_CHANNELS: int = 64
    MAXPOOL_TILED_OUT_HEIGHT: int = 32
    MAXPOOL_TILED_OUT_WIDTH: int = 32
    MAXPOOL_TILED_CHANNELS: int = 128
    LAYERNORM_TILED_N: int = 1024
    BATCHNORM_TILED_CHANNELS: int = 64
    BATCHNORM_TILED_HEIGHT: int = 64
    BATCHNORM_TILED_WIDTH: int = 64
    GEMM_TILED_OUT_DIM: int = 512
    GEMM_TILED_IN_DIM: int = 512
    GEMM_TILED_N: int = 128
    MULTIHEAD_ATTENTION_TILED_Q_LEN: int = 32
    MULTIHEAD_ATTENTION_TILED_KV_LEN: int = 32
    ADD_TILED_SIZE: int = 4096
    SIGMOID_TILED_SIZE: int = 4096

    @property
    def CDEFS(self): # Generate C preprocessor definitions
        return " ".join([f"-D{key}={value}" for key, value in self.__dict__.items()])

    @property
    def hash(self) -> str:
        """
        Generate a unique hash for the configuration based on its parameters.
        This can be used to identify unique configurations in a database or cache.
        """
        items = sorted(self.__dict__.items())
        config_str = "_".join(f"{k}={v}" for k, v in items)
        return hashlib.md5(config_str.encode()).hexdigest()

    def to_json(self, file_path: str) -> None:
        """
        Save the parameters to a JSON file.
        """
        with open(file_path, 'w') as json_file:
            json.dump(self.__dict__, json_file, indent=4)

    @classmethod
    def from_json(cls, file_path: str) -> 'SramTileParameters':
        """
        Load the parameters from a JSON file.
        """
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return cls(**data)