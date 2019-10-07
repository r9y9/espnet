#!/usr/bin/env python
#  coding: utf-8
<<<<<<< HEAD
"""Feats to npy conversion.

usage: feats2npy.py [options] <scp> <out_dir>

options:
    -h, --help               Show help message.
"""
from docopt import docopt
import sys
import numpy as np
from os.path import join
from kaldiio import ReadHelper
import os

if __name__ == "__main__":
    args = docopt(__doc__)
    scp_file = args["<scp>"]
    out_dir = args["<out_dir>"]
    os.makedirs(out_dir, exist_ok=True)
    with ReadHelper(f"scp:{scp_file}") as f:
        for utt_id, arr in f:
            out_path = join(out_dir, f"{utt_id}-feats.npy")
=======

import argparse
from kaldiio import ReadHelper
import numpy as np
import os
from os.path import join
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description='Convet kaldi-style features to numpy arrays',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('scp_file', type=str, help='scp file')
    parser.add_argument('out_dir', type=str, help='output directory')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    os.makedirs(args.out_dir, exist_ok=True)
    with ReadHelper(f"scp:{args.scp_file}") as f:
        for utt_id, arr in f:
            out_path = join(args.out_dir, f"{utt_id}-feats.npy")
>>>>>>> origin/master
            np.save(out_path, arr, allow_pickle=False)
    sys.exit(0)
