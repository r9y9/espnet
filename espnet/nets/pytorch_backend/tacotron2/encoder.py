#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron2 encoder related modules."""

import six

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def _encoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('relu'))


class EncoderEmbedding(torch.nn.Module):
    def __init__(self, idim, embed_dim, padding_idx=0):
        super(EncoderEmbedding, self).__init__()
        self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        self.odim = embed_dim

    def get_odim(self):
        return self.odim

    def forward(self, xs):
        return self.embed(xs)


class Prenet(torch.nn.Module):
    def __init__(self, idim,
                 conv_layers=3,
                 conv_chans=512,
                 conv_filts=5,
                 use_residual=False,
                 dropout_rate=0.5):
        super(Prenet, self).__init__()
        self.odim = conv_chans
        self.use_residual = use_residual

        assert conv_layers > 0
        self.conv_blocks = torch.nn.ModuleList()
        for layer in range(conv_layers):
            ichans = idim if layer == 0 else conv_chans
            self.conv_blocks += [torch.nn.Sequential(
                torch.nn.Conv1d(ichans, conv_chans, conv_filts, stride=1,
                                padding=(conv_filts - 1) // 2, bias=False),
                torch.nn.BatchNorm1d(conv_chans),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate))]

        self.apply(_encoder_init)

    def get_odim(self):
        return self.odim

    def forward(self, xs):
        for conv in self.conv_blocks:
            if self.use_residual:
                xs += conv(xs)
            else:
                xs = conv(xs)
        return xs


class EncoderBody(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Tacotron2, which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_. This is the encoder which converts the
    sequence of characters into the sequence of hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, layers=2, units=512, dropout_rate=0.5):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            layers (int, optional) The number of encoder blstm layers.
            units (int, optional) The number of encoder blstm units.
            dropout_rate (float, optional) Dropout rate.

        """
        super(EncoderBody, self).__init__()

        if layers > 0:
            self.blstm = torch.nn.LSTM(
                idim, units // 2, layers,
                batch_first=True,
                bidirectional=True)
        else:
            self.blstm = None

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence of character ids (B, Tmax). Padded value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, units).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        if self.blstm is None:
            # TODO: test this. probably check axis
            return xs.transpose(1, 2)

        xs = pack_padded_sequence(xs.transpose(1, 2), ilens, batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        # TODO: should be okay to ignore returned hlens, but just in case check this again later
        xs, _ = pad_packed_sequence(xs, batch_first=True)

        return xs, ilens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,).

        Returns:
            Tensor: The sequences of encoder states(T, units).

        """
        # TODO: is this really needed?
        assert len(x.size()) == 1
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0][0]
