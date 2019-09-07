#!/usr/bin/env python
#  coding: utf-8
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
            np.save(out_path, arr, allow_pickle=False)
    sys.exit(0)
