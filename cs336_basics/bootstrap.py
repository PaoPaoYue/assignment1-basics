import argparse
from dataclasses import dataclass, fields
import enum
import logging
import os

import numpy as np
import torch
import wandb

from cs336_basics.bpe import *
from cs336_basics.tokenizer import *
from cs336_basics.train import *
from cs336_basics.utils import dict_to_params

logger = logging.getLogger()


class Command(enum.Enum):
    BPE = "train_bpe"
    TOKENIZE = "tokenize"
    TRAIN = "train_model"


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for CS336")
    parser.add_argument("--log_level", type=str, default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ===== BPE =====
    bpe_parser = subparsers.add_parser("train_bpe", help="Train BPE tokenizer")
    bpe_parser.add_argument("--input_path", type=str, required=True, help="Path to input text file")
    bpe_parser.add_argument("--vocab_size", type=int, default=10000, help="Size of vocabulary")
    bpe_parser.add_argument("--special_tokens", type=str, nargs="+", default=['<|endoftext|>'], help="List of special tokens")
    bpe_parser.add_argument("--num_processors", type=int, default=1, help="Number of processes to use")
    bpe_parser.add_argument("--out_dir_path", type=str, required=True, help="Directory to save vocabulary and merges")

    # ===== TOKENIZE =====
    tok_parser = subparsers.add_parser("tokenize", help="Tokenize dataset")
    tok_parser.add_argument("--input_path", type=str, required=True, help="Path to input text file")
    tok_parser.add_argument("--vocab_dir_path", type=str, required=True, help="Path to vocabulary directory")
    tok_parser.add_argument("--special_tokens", type=str, nargs="+", default=['<|endoftext|>'], help="List of special tokens")
    tok_parser.add_argument("--chunk_size", type=int, default=1048576, help="Size of each processing buffer chunk")
    tok_parser.add_argument("--num_processors", type=int, default=1, help="Number of processes to use")
    tok_parser.add_argument("--out_dir_path", type=str, required=True, help="Directory to save tokenized output")

    # ===== TRAIN =====
    train_parser = subparsers.add_parser("train_model", help="Train language model")
    train_parser.add_argument("--train_dir_path", type=str, required=True)
    train_parser.add_argument("--valid_dir_path", type=str, required=True)
    train_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to save model checkpoints")

    train_parser.add_argument("--seed", type=int, default=42)

    train_parser.add_argument("--vocab_size", type=int, default=10000)
    train_parser.add_argument("--context_length", type=int, default=256)
    train_parser.add_argument("--d_model", type=int, default=512)
    train_parser.add_argument("--d_ff", type=int, default=1344)
    train_parser.add_argument("--num_layers", type=int, default=4)
    train_parser.add_argument("--num_heads", type=int, default=16)
    train_parser.add_argument("--rope_theta", type=float, default=10000.0)

    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--loader_num_workers", type=int, default=8)
    train_parser.add_argument("--max_norm", type=float, default=0)
    train_parser.add_argument("--optimizer_beta1", type=float, default=0.9)
    train_parser.add_argument("--optimizer_beta2", type=float, default=0.999)
    train_parser.add_argument("--optimizer_weight_decay", type=float, default=0.01)
    train_parser.add_argument("--scheduler_t", type=int, default=5)
    train_parser.add_argument("--scheduler_t_mult", type=int, default=1)
    train_parser.add_argument("--scheduler_min_lr", type=float, default=0.0)

    train_parser.add_argument("--num_epochs", type=int, default=10)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--min_delta", type=float, default=1e-4)

    train_parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == Command.BPE.value:
        params = dict_to_params(args, BpeParams)
    elif args.command == Command.TOKENIZE.value:
        params = dict_to_params(args, TokenizeParams)
    elif args.command == Command.TRAIN.value:
        params = dict_to_params(args, TrainParams)
    else:
        parser.error(f"Unknown command: {args.command}")

    return params, args

if __name__ == "__main__":
    params, args = parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Starting Task: {args.command}")
    logger.info(f"Parameters: {params}")

    if args.command == Command.BPE.value:
        train_bpe(params)
    elif args.command == Command.TOKENIZE.value:
        tokenize(params)
    elif args.command == Command.TRAIN.value:
        train_model(params)