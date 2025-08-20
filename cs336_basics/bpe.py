import collections
from dataclasses import dataclass
import pickle
import os
from tqdm import tqdm

from cs336_basics.pretokenization import pretokenize

import logging
logger = logging.getLogger(__name__)

@dataclass
class BpeParams:
    input_path: str
    vocab_size: int
    special_tokens: list[str]
    num_processors: int
    out_dir_path: str

def train_bpe(
    params: BpeParams,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
      input_path (str | os.PathLike): Path to BPE tokenizer training data.
      vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
      special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        These strings will never be split into multiple tokens, and will always be
        kept as a single token. If these special tokens occur in the `input_path`,
        they are treated as any other string.

    Returns:
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab:
          The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
          to bytes (token bytes)
        merges:
          BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
          representing that <token1> was merged with <token2>.
          Merges are ordered by order of creation.
    """
    pretoken_byte_counts = pretokenize(
        input_path=params.input_path,
        special_tokens=params.special_tokens,
        num_processors=params.num_processors
    )

    vocab_list = [token.encode("utf-8") for token in params.special_tokens] + [bytes([i]) for i in range(256)]

    logger.info("Determining merges")
    merges = determine_merges(pretoken_byte_counts, params.vocab_size - len(vocab_list))
    logger.info("Done determining merges")

    vocab_list.extend([b"".join((a, b)) for a, b in merges])

    vocab = {i: token for i, token in enumerate(vocab_list)}

    if params.out_dir_path:
        os.makedirs(params.out_dir_path, exist_ok=True)
        with open(f"{params.out_dir_path}/vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
            logger.info("Saved vocabulary to %s", f"{params.out_dir_path}/vocab.pkl")
        with open(f"{params.out_dir_path}/merges.pkl", "wb") as f:
            pickle.dump(merges, f)
            logger.info("Saved merges to %s", f"{params.out_dir_path}/merges.pkl")
    return vocab, merges

def determine_merges(
    pretoken_byte_counts: collections.Counter[tuple[bytes]], 
    merge_token_allowance: int
) -> list[tuple[bytes, bytes]]:
    """
    增量更新版 BPE 训练
    """
    merges = []
    # 初始 pair_counts
    pair_counts = collections.Counter()
    token_to_pairs = {}  # token -> set of pairs
    for token, freq in pretoken_byte_counts.items():
        pairs = collections.defaultdict(int)
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += freq
            pairs[pair] += 1
        token_to_pairs[token] = pairs

    for _ in tqdm(range(merge_token_allowance), desc="BPE merges", total=merge_token_allowance):
        if not pair_counts:
            break

        # 找到出现次数最多的 pair
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        # 找到所有包含 best_pair 的 token
        affected_tokens = [token for token in pretoken_byte_counts if best_pair in token_to_pairs[token]]

        # 更新这些 token
        for token in affected_tokens:
            freq = pretoken_byte_counts[token]
            new_token = []
            i = 0
            while i < len(token):
                if (
                    i < len(token) - 1 
                    and token[i] == best_pair[0] 
                    and token[i + 1] == best_pair[1]
                ):
                    new_token.append(token[i] + token[i + 1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1

            new_token = tuple(new_token)

            # 从全局 pair_counts 中减去旧 pairs
            for pair, count in token_to_pairs[token].items():
                pair_counts[pair] -= freq * count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # 替换 token
            del pretoken_byte_counts[token]
            pretoken_byte_counts[new_token] += freq

            # 计算新 token 的 pairs
            new_pairs = collections.defaultdict(int)
            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i + 1])
                pair_counts[pair] += freq
                new_pairs[pair] += 1
            token_to_pairs[new_token] = new_pairs
    return merges