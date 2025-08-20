from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import Pool, Manager, Queue
import pickle
import logging
import os
import tempfile
from threading import Thread
from typing import Optional, Dict, List, Tuple, Iterator, Iterable
import re
import json
import numpy as np
import regex
from tqdm import tqdm
import zarr

from cs336_basics.pretokenization import PRETOKEN_PATTERN, BYTE_CACHE, find_chunk_boundaries
from cs336_basics.utils import zarrs_1d_to_npy

import logging
logger = logging.getLogger(__name__)

@dataclass
class TokenizeParams:
    input_path: str
    vocab_dir_path: str
    special_tokens: list[str]
    chunk_size: int
    num_processors: int
    out_dir_path: str

def tokenize(params: TokenizeParams) -> None:
    os.makedirs(params.out_dir_path, exist_ok=True)

    with open(params.input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            params.num_processors,
            [t.encode("utf-8") for t in params.special_tokens],
        )
    with tempfile.TemporaryDirectory(dir=f"{params.out_dir_path}/") as tmpdir:
        with Manager() as manager:
            queue = manager.Queue()
            chunk_args = [
                ProcessChunkArgs(start, 
                                 end, 
                                 params.input_path,  
                                 params.vocab_dir_path, 
                                 params.special_tokens, 
                                 params.chunk_size, 
                                 queue, 
                                 tmpdir)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            Thread(
                target=progress_listener, 
                args=(queue,  os.path.getsize(params.input_path)), 
                daemon=True
            ).start()
            with Pool(params.num_processors) as pool:
                tmp_arrs = [tmp_arr for tmp_arr in pool.imap(process_chunk, chunk_args)]
                queue.put(-1) # notify listener completion

        logger.info("Finished tokenizing all chunks, concatenating results")
        zarrs_1d_to_npy(tmp_arrs, f"{params.out_dir_path}/tokens.npy")

    logger.info("Tokenized file %s and saved to %s", params.input_path, params.out_dir_path)

__BATCH_SIZE = 1048576 # 4mb
__REPORT_INTERVAL = 4096

@dataclass
class ProcessChunkArgs:
    start: int
    end: int
    input_path: str | os.PathLike
    vocab_dir_path: str
    special_tokens: list[str]
    chunk_size: int
    queue: Queue
    tmpdir: str

def process_chunk(args: ProcessChunkArgs) -> zarr.Array:
    with open(f"{args.vocab_dir_path}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"{args.vocab_dir_path}/merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tokenizer = BpeTokenizer(vocab, merges, special_tokens=args.special_tokens)

    tmp_arr = zarr.open_array(store=zarr.DirectoryStore(f"{args.tmpdir}/chunk_{args.start}_{args.end}.zarr"),
                               mode="w", shape=(0,), chunks=(args.chunk_size,), dtype=np.int32)
    with open(args.input_path, "rb") as f:
        f.seek(args.start)
        buffer = np.zeros(args.chunk_size, dtype=np.int32)
        buffer_index = 0
        cur = args.start
        for token in tokenizer.encode_iterable_bytes(f):
            buffer[buffer_index] = token
            buffer_index += 1
            if buffer_index == args.chunk_size:
                tmp_arr.append(buffer)
                buffer_index = 0
            
            next = f.tell()
            if next >= args.end:
                break
            if buffer_index % __REPORT_INTERVAL == 0:
                args.queue.put(next-cur)
                cur = next

        if buffer_index > 0:
            tmp_arr.append(buffer[:buffer_index])
    return tmp_arr

def progress_listener(queue: Queue, total: int):
    with tqdm(desc="Tokenizing", total=total, unit="B", unit_scale=True) as pbar:
        while True:
            progress = queue.get()
            if progress < 0:
                break
            pbar.update(progress)

class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        """
        vocab: id -> token_bytes（必须包含所有可用 token 的最终字节序列，含特殊符号）
        merges: 已训练得到的合并对（此实现中仅保存，不参与编码计算）
        special_tokens: 形如 "<endoftext>" 的字面字符串列表（可为空）
        """
        self.id_to_bytes: Dict[int, bytes] = dict(vocab)
        # 反向映射：bytes -> id
        self.bytes_to_id: Dict[bytes, int] = {b: i for i, b in self.id_to_bytes.items()}
        self.merge_priority: Dict[Tuple[bytes, bytes], int] = {
            pair: idx for idx, pair in enumerate(merges)
        }

        # 预处理 special tokens
        self.special_tokens: List[str] = list(special_tokens or [])
        # 映射 special string -> id（通过 utf-8 编码后的 bytes 查 id）
        self.special_to_id: Dict[bytes, int] = {}
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b not in self.bytes_to_id:
                raise ValueError(f"Special token {s!r} not found in vocab as bytes.")
            self.special_to_id[b] = self.bytes_to_id[b]

        # 为编码阶段预编译一个用于切分 special 的正则（最长优先）
        if self.special_tokens:
            # 优先匹配更长的 special，防止前缀截断
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(s) for s in specials_sorted)
            self.special_tokens_pattern = re.compile(pattern.encode("utf-8"))
        else:
            self.special_tokens_pattern = None

    @classmethod
    def from_files(vocab_filepath, merges_filepath, special_tokens=None):
        """
        从文件加载 vocab 和 merges。
        vocab_filepath: json 文件路径
        merges_filepath: json 文件路径
        special_tokens: 可选特殊符号列表
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = json.load(f)
        return BpeTokenizer(vocab, merges, special_tokens)

    def _encode_bytes_with_trie(self, data: bytes) -> List[int]:
        ids: List[int] = []
        text = data.decode("utf-8", errors="replace")
        # 使用 finditer 实现流式匹配
        for match in regex.finditer(PRETOKEN_PATTERN, text):
            tokens: List[bytes] = [BYTE_CACHE[b] for b in match.group(0).encode("utf-8")]

            while True:
                # 查找当前轮次中所有可合并的 pair，并记录最优优先级
                best_priority = float("inf")
                best_candidates: List[int] = []

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_priority:
                        priority = self.merge_priority[pair]
                        if priority < best_priority:
                            best_priority = priority
                            best_candidates = {i}
                        elif priority == best_priority:
                            best_candidates.add(i)

                if not best_candidates:
                    break

                # 从左到右合并所有最佳 pair，避免重叠
                new_tokens: List[bytes] = []
                i = 0
                while i < len(tokens):
                    # 尝试合并当前 pair
                    if i in best_candidates and i + 1 < len(tokens):
                        merged = tokens[i] + tokens[i + 1]
                        new_tokens.append(merged)
                        i += 2  # 跳过下一个位置，避免重叠
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                tokens = new_tokens

            # 将最终 token 映射为 vocab 中的 id

            for token in tokens:
                if token not in self.bytes_to_id:
                    raise ValueError(f"Token {token!r} not found in vocab.")
                ids.append(self.bytes_to_id[token])

        return ids


    def encode(self, text: str) -> List[int]:
        return self.encode_bytes(text.encode("utf-8", errors="replace"))
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for data in iterable:
            if not data:
                continue
            for tid in self.encode(data):
                yield tid

    def encode_bytes(self, text: bytes) -> List[bytes]:
        if not text:
            return []

        if not self.special_tokens_pattern:
            return self._encode_bytes_with_trie(text)

        ids: List[int] = []
        pos = 0
        for m in self.special_tokens_pattern.finditer(text):
            start, end = m.span()
            # 先处理 special 前面的普通片段
            if start > pos:
                chunk = text[pos:start]
                ids.extend(self._encode_bytes_with_trie(chunk))

            # 处理 special 本身
            ssp = m.group(0)
            tid = self.special_to_id.get(ssp)
            if tid is None:
                # 若未声明为 special（或不在 vocab），回退成普通文本处理
                # 一般不会触发，因为 __init__ 已校验 special 均在 vocab
                ids.extend(self._encode_bytes_with_trie(ssp))
            else:
                ids.append(tid)

            pos = end

        # 处理最后的尾部普通片段
        if pos < len(text):
            tail = text[pos:]
            ids.extend(self._encode_bytes_with_trie(tail))

        return ids

    def encode_iterable_bytes(self, iterable: Iterable[bytes]) -> Iterator[int]:
        """
        流式编码：对一个字节可迭代对象逐个编码并按顺序产出 token id。
        - 与 encode_bytes(text) 的切分逻辑一致（包含 special_tokens 处理）
        - 不会在元素之间插入任何分隔符
        - 不跨元素匹配 special token
        """
        if self.special_tokens_pattern is None:
            # 无 special，直接按块进行字节级最长匹配
            for data in iterable:
                if not data:
                    continue
                for tid in self._encode_bytes_with_trie(data):
                    yield tid
            return

        # 有 special，逐块在字节层面先切分 special，再进行字节级匹配
        for data in iterable:
            if not data:
                continue

            pos = 0
            for m in self.special_tokens_pattern.finditer(data):
                start, end = m.span()
                # 先处理 special 前面的普通片段
                if start > pos:
                    chunk = data[pos:start]
                    for tid in self._encode_bytes_with_trie(chunk):
                        yield tid

                # 处理 special 本身
                ssp = m.group(0)
                tid = self.special_to_id.get(ssp)
                if tid is None:
                    # 回退为普通文本处理（理论上不会触发，__init__ 已校验）
                    for _tid in self._encode_bytes_with_trie(
                        ssp.encode("utf-8", errors="strict")
                    ):
                        yield _tid
                else:
                    yield tid

                pos = end

            # 处理尾部普通片段
            if pos < len(data):
                tail = data[pos:]
                for tid in self._encode_bytes_with_trie(tail):
                    yield tid

    def decode(self, ids: List[int]) -> str:
        """
        将 token id 列表还原为字符串。
        - 先拼接 bytes，再用 UTF-8 解码；
        - 使用 errors="replace" 以容忍任何非标准字节序列，避免解码异常。
        """
        if ids is None or len(ids) < 0:
            return ""
        try:
            data = b"".join(self.id_to_bytes[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"Unknown token id in sequence: {e!r}")

        # 容错解码：遇到非法 UTF-8 序列，用 U+FFFD 替换
        return data.decode("utf-8", errors="replace")
