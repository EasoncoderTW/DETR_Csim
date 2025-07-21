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


class ModelWeight(object):
    DATA_TYPE = {
        "fp32": 0,
        "int8": 1,
    }

    PACK_MATHED = {
        "normal": 0,
        "qpack": 1,
    }

    def __init__(self, torch_model, dict_version = 1,**kargs):
        # torch_model: torch.nn.Module or dict
        if isinstance(torch_model, torch.nn.Module):
            # If it's a torch model, get the state_dict
            self.model_state_dict = torch_model.state_dict()
        elif isinstance(torch_model, dict):
            # If it's a dict, use it directly
            self.model_state_dict = torch_model
        else:
            raise TypeError("torch_model must be a torch.nn.Module or a dict")
        self.version = dict_version
        self.pack_method = kargs.get('pack_method', "normal")
        self.data_type = kargs.get('data_type', "fp32")

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
        other_count = 0
        tensor_size = 0
        for param_tensor in self.model_state_dict:
            name = param_tensor.split('.')
            tensor = self.model_state_dict[param_tensor]
            if name[-1] == "weight":
                weight_count+=1
            elif name[-1] == "bias":
                bias_count+=1
            else:
                other_count+=1
            tensor_size += tensor.size().numel()
                # print(param_tensor, "\t", tensor.size())

        print(f"[total weight count]", weight_count)
        print(f"[total bias count]", bias_count)
        print(f"[total others count]", other_count)
        print(f"[total tensor size] ",self.format_bytes(tensor_size*4))

    def save_state_dict(self, filename, callback: Callable[[str,torch.Tensor], bytes]):
        head = bytes()
        table = bytes()
        name = bytes()
        data = bytes()

        tensor_count = 0
        for tensor_name in tqdm(self.model_state_dict, "Processing Weight "):
            tensor = self.model_state_dict[tensor_name]

            # pack into bytes
            tensor_name_Cstr_bytes = bytes(tensor_name+'\0', 'ascii')
            data_bytes = callback(tensor_name, tensor) # custom pack method

            info = TensorInfo(
                        id=tensor_count,
                        data_type = ModelWeight.DATA_TYPE[self.data_type],
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
        head += struct.pack('I',ModelWeight.PACK_MATHED[self.pack_method]) # packmethod
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
def normal_pack(tensor_name:str, tensor:torch.Tensor)->bytes:
    # Convert to float32
    tensor = tensor.to(dtype=torch.float32)

    if tensor_name == "query_embed.weight":
        tensor = tensor.transpose(0,1)
        #print(tensor_name, tensor.shape)

    # print(tensor_name, tensor.shape)

    # Convert to bytes
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()

    return tensor_bytes

import argparse

def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Export model weights to binary", add_help=add_help)

    parser.add_argument(
        "-r", "--repo_or_dir",
        type=str,
        default="facebookresearch/detr:main",
        help="Repo name (e.g. 'facebookresearch/detr:main') or local dir"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="detr_resnet50",
        help="Model name (e.g. 'detr_resnet50')"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output binary file"
    )

    parser.add_argument(
        "-l", "--list",
        type=bool,
        required=False,
        help="List model weights"
    )

    parser.add_argument(
        "-p", "--peek",
        type=bool,
        default=False,
        required=False,
        help="Peek at the model weights without saving"
    )
    return parser

def run_model_weight(args):
    print("Repo or dir:", args.repo_or_dir)
    print("Model:", args.model)
    print("Output path:", args.output)
    print("List model weights:", args.list)

    model = torch.hub.load(args.repo_or_dir, args.model, pretrained=True)
    MD = ModelWeight(model)
    if args.list:
        print("Model weights:")
        for name, tensor in MD.model_state_dict.items():
            print(f"{name}: {tensor.shape}")
    MD.summary()

    if not args.peek:
        MD.save_state_dict(args.output, normal_pack)


def main(args):
    run_model_weight(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)