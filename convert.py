import argparse
from pathlib import Path
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == '__main__':
    ## read a path using argparse and pass it to convert_zero_checkpoint_to_fp32_state_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='path to the desired checkpoint folder')
    parser.add_argument('--output', type=str, default=None, help='path to the pytorch fp32 state_dict output file')
    # parser.add_argument('--tag', type=str, help='checkpoint tag used as a unique identifier for checkpoint')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(args.input) / 'converted.ckpt'
    convert_zero_checkpoint_to_fp32_state_dict(args.input, args.output)