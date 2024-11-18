import sys
import os
import torch
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


def convert_deepspeed_to_pytorch(deepspeed_checkpoint_dir, output_path):
    """
    Convert a DeepSpeed checkpoint to a single PyTorch .pt file
    Args:
        deepspeed_checkpoint_dir (str): Directory containing DeepSpeed checkpoint files
        output_path (str): Desired output path for the .pt file
    """
    # Create temporary directory for intermediate files
    temp_dir = output_path + "_temp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # First convert to intermediate format
        print(
            f"Converting DeepSpeed checkpoint from directory: {deepspeed_checkpoint_dir}"
        )
        state_dict = convert_zero_checkpoint_to_fp32_state_dict(
            deepspeed_checkpoint_dir, temp_dir
        )

        # Load all parts if multiple files were created
        if os.path.isfile(os.path.join(temp_dir, "pytorch_model.bin")):
            # Single file case
            state_dict = torch.load(os.path.join(temp_dir, "pytorch_model.bin"))
        else:
            # Multiple file case
            state_dict = {}
            for filename in sorted(os.listdir(temp_dir)):
                if filename.startswith("pytorch_model-") and filename.endswith(".bin"):
                    part_dict = torch.load(os.path.join(temp_dir, filename))
                    state_dict.update(part_dict)

        # Save as single .pt file
        print(f"Saving converted checkpoint to: {output_path}")
        torch.save(state_dict, output_path)
        print("Conversion complete!")

    finally:
        # Cleanup temporary directory
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Ensure two arguments are passed (the checkpoint directory and output filename)
    if len(sys.argv) != 3:
        print(
            "Usage: python deepspeed_to_pytorch.py <deepspeed_checkpoint_dir> <output_filename>"
        )
        sys.exit(1)

    deepspeed_checkpoint_dir = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isdir(deepspeed_checkpoint_dir):
        print(
            f"Error: The checkpoint directory '{deepspeed_checkpoint_dir}' does not exist."
        )
        sys.exit(1)

    # Run the conversion function
    convert_deepspeed_to_pytorch(deepspeed_checkpoint_dir, output_path)
