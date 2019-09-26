#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules."""

import logging

from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F

from espnet.utils.dynamic_import import dynamic_import
import yaml

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHG
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHGLoss
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.fill_missing_args import fill_missing_args


def _instantiate(args):
    name = args.pop("module")
    return dynamic_import(name)(**args)


class MultiSpeakerAttSeq2SeqModel(TTSInterface, torch.nn.Module):
    """A generic class for attention-based end-to-end TTS model.
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("tacotron 2 model setting")
        group.add_argument('--model-config', type=str, help='Type of attention mechanism')
        return parser

    def __init__(self, idim, odim, args=None):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        with open(args.model_config) as f:
            model_args = yaml.load(f, Loader=yaml.Loader)

        # fill missing arguments
        # TODO: do we really need this?
        # args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = idim
        self.odim = odim

        # Input embedding
        self.text_embedding = _instantiate(model_args["input_embedding"]["text"])

        # Encoder
        self.encoder_prenet = _instantiate(model_args["encoder"]["prenet"])
        self.encoder_body = _instantiate(model_args["encoder"]["body"])

        # Decoder
        self.decoder_prenet = _instantiate(model_args["decoder"]["prenet"])
        self.decoder_body = _instantiate(model_args["decoder"]["body"])

        import ipdb
        ipdb.set_trace()

        self.attention = _instantiate(model_args.attention)

        self.stoptoken_predictor = _instantiate(model_args.stoptoken_predictor)

    def forward(self, xs, ilens, ys, labels, olens, spembs=None, spcs=None, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).
            spcs (Tensor, optional): Batch of groundtruth spectrograms (B, Lmax, spc_dim).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        max_in = max(ilens)
        max_out = max(olens)
        if max_in != xs.shape[1]:
            xs = xs[:, :max_in]
        if max_out != ys.shape[1]:
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]

        # calculate tacotron2 outputs
        hs, hlens = self.enc(xs, ilens)
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # caluculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs, before_outs, logits, ys, labels, olens)
        loss = l1_loss + mse_loss + bce_loss
        report_keys = [
            {'l1_loss': l1_loss.item()},
            {'mse_loss': mse_loss.item()},
            {'bce_loss': bce_loss.item()},
        ]

        # caluculate attention loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive input will be changed when r > 1
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens, olens_in)
            loss = loss + attn_loss
            report_keys += [
                {'attn_loss': attn_loss.item()},
            ]

        # caluculate cbhg loss
        if self.use_cbhg:
            # remove unnecessary padded part (for multi-gpus)
            if max_out != spcs.shape[1]:
                spcs = spcs[:, :max_out]

            # caluculate cbhg outputs & loss and report them
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            cbhg_l1_loss, cbhg_mse_loss = self.cbhg_loss(cbhg_outs, spcs, olens)
            loss = loss + cbhg_l1_loss + cbhg_mse_loss
            report_keys += [
                {'cbhg_l1_loss': cbhg_l1_loss.item()},
                {'cbhg_mse_loss': cbhg_mse_loss.item()},
            ]

        report_keys += [{'loss': loss.item()}]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, spemb=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        """
        # get options
        threshold = inference_args.threshold
        minlenratio = inference_args.minlenratio
        maxlenratio = inference_args.maxlenratio

        # inference
        h = self.enc.inference(x)
        if self.spk_embed_dim is not None:
            spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        outs, probs, att_ws = self.dec.inference(h, threshold, minlenratio, maxlenratio)

        if self.use_cbhg:
            cbhg_outs = self.cbhg.inference(outs)
            return cbhg_outs, probs, att_ws
        else:
            return outs, probs, att_ws

    def calculate_all_attentions(self, xs, ilens, ys, spembs=None, *args, **kwargs):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            numpy.ndarray: Batch of attention weights (B, Lmax, Tmax).

        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        self.eval()
        with torch.no_grad():
            hs, hlens = self.enc(xs, ilens)
            if self.spk_embed_dim is not None:
                spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
                hs = torch.cat([hs, spembs], dim=-1)
            att_ws = self.dec.calculate_all_attentions(hs, hlens, ys)
        self.train()

        return att_ws.cpu().numpy()

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ['loss', 'l1_loss', 'mse_loss', 'bce_loss']
        if self.use_guided_attn_loss:
            plot_keys += ['attn_loss']
        if self.use_cbhg:
            plot_keys += ['cbhg_l1_loss', 'cbhg_mse_loss']
        return plot_keys
