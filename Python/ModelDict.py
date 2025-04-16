from typing import Any, Callable
import torch
import struct
from tqdm import tqdm

BAR_LEN = 30

class TensorInfo(dict):
    FORMAT = '<6I'  # Little-endian, 6 unsigned 32-bit integers

    def __init__(self, id=0, data_type=0, data_offset=0, data_size=0, name_offset=0, name_size=0):
        super().__init__(
            id=id,
            data_type=data_type,
            data_offset=data_offset,
            data_size=data_size,
            name_offset=name_offset,
            name_size=name_size
        )

    def to_bytes(self):
        return struct.pack(
            self.FORMAT,
            self['id'],
            self['data_type'],
            self['data_offset'],
            self['data_size'],
            self['name_offset'],
            self['name_size']
        )

    @classmethod
    def from_bytes(cls, byte_data):
        unpacked = struct.unpack(cls.FORMAT, byte_data)
        return cls(*unpacked)


class ModelDict(object):
    DATA_TYPE = {
        "fp32": 0,
    }

    PACK_MATHED = {
        "normal": 0,
    }

    def __init__(self, torch_model, dict_version = 1,**kargs):
        self.model = torch_model.cpu()
        self.version = dict_version
        self.kargs = kargs

    def format_bytes(self, size):
        # 2**10 = 1024
        power = 1024
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        i = 0
        while size >= power and i < len(units) - 1:
            size /= power
            i += 1
        return f"{size:.2f} {units[i]}"

    def summary(self):
        weight_count = 0
        bias_count = 0
        tensor_size = 0
        for param_tensor in self.model.state_dict():
            name = param_tensor.split('.')
            if name[-1] in ["weight","bias"]:
                tensor = self.model.state_dict()[param_tensor]
                if name[-1] == "weight":
                    weight_count+=1
                if name[-1] == "bias":
                    bias_count+=1
                tensor_size += tensor.size().numel()
                # print(param_tensor, "\t", tensor.size())

        print(f"[total weight count]", weight_count)
        print(f"[total bias count]", bias_count)
        print(f"[total tensor size] {tensor_size * 4/1024/1024:.3f} MB")

    def save_state_dict(self, filename, callback: Callable[[torch.Tensor], bytes]):
        head = bytes()
        table = bytes()
        name = bytes()
        data = bytes()

        tensor_count = 0
        for tensor_name in tqdm(self.model.state_dict(), "Processing Weight "):
            is_weight = (tensor_name[-6:] == "weight")
            is_bias = (tensor_name[-4:] == "bias")
            if is_weight or is_bias:
                tensor = self.model.state_dict()[tensor_name]

                # pack into bytes
                tensor_name_Cstr_bytes = bytes(tensor_name+'\0', 'ascii')
                data_bytes = callback(tensor) # custom pack method

                info = TensorInfo(
                            id=tensor_count,
                            data_type = ModelDict.DATA_TYPE['fp32'],
                            data_offset = len(data),
                            data_size = len(data_bytes),
                            name_offset = len(name),
                            name_size = len(tensor_name_Cstr_bytes))

                tensor_count+=1

                # concat
                table += info.to_bytes()
                name += tensor_name_Cstr_bytes # C string
                data += data_bytes


        HEAD_SIZE = 24
        head += struct.pack('I',self.version) # version
        head += struct.pack('I',tensor_count) # tensor count
        head += struct.pack('I',ModelDict.PACK_MATHED['normal']) # packmethod
        head += struct.pack('I',HEAD_SIZE) # info offset
        head += struct.pack('I',HEAD_SIZE+len(table)) # name offset
        head += struct.pack('I',HEAD_SIZE+len(table)+len(name)) # data offset

        # concat
        output_bytes = head + table + name + data

        with open(filename, "wb") as f:
            write_out = f.write(output_bytes)

        print("="*BAR_LEN)
        print(f"Saved state dict to {filename} ({self.format_bytes(write_out)})")
        print(f"- Head  : {self.format_bytes(len(head))}")
        print(f"- Table : {self.format_bytes(len(table))}")
        print(f"- Name  : {self.format_bytes(len(name))}")
        print(f"- Data  : {self.format_bytes(len(data))}")
        print(f"- Total : {self.format_bytes(len(output_bytes))}")
        print("="*BAR_LEN)


#########
def normal_pack(tensor:torch.Tensor)->bytes:
    # Convert to float32
    tensor = tensor.to(dtype=torch.float32)

    # Convert to bytes
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()

    return tensor_bytes