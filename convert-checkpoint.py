import sys
import os
import torch
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

def convert_deepspeed_to_pytorch(deepspeed_checkpoint_dir, output_path):
    # Convert the DeepSpeed ZeRO checkpoint to a single FP32 PyTorch state_dict
    print(f"Converting DeepSpeed checkpoint from directory: {deepspeed_checkpoint_dir}")
    model_state_dict = convert_zero_checkpoint_to_fp32_state_dict(deepspeed_checkpoint_dir, output_path)

    print(f"Saving converted checkpoint to: {output_path}")
    print("Conversion complete!")

if __name__ == "__main__":
    # Ensure two arguments are passed (the checkpoint directory and output filename)
    if len(sys.argv) != 3:
        print("Usage: python deepspeed_to_pytorch.py <deepspeed_checkpoint_dir> <output_filename>")
        sys.exit(1)

    deepspeed_checkpoint_dir = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isdir(deepspeed_checkpoint_dir):
        print(f"Error: The checkpoint directory '{deepspeed_checkpoint_dir}' does not exist.")
        sys.exit(1)

    # Run the conversion function
    convert_deepspeed_to_pytorch(deepspeed_checkpoint_dir, output_path)

