import json
from typing import Optional, Dict, List, Tuple, Iterator, Iterable
import re
import regex

from cs336_basics.pretokenization import PRETOKEN_PATTERN, BYTE_CACHE

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
        self.special_to_id: Dict[str, int] = {}
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b not in self.bytes_to_id:
                raise ValueError(f"Special token {s!r} not found in vocab as bytes.")
            self.special_to_id[s] = self.bytes_to_id[b]

        # 为编码阶段预编译一个用于切分 special 的正则（最长优先）
        if self.special_tokens:
            # 优先匹配更长的 special，防止前缀截断
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(s) for s in specials_sorted)
            self.special_tokens_pattern = re.compile(pattern)
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
        """
        将文本编码为 token id 列表。
        """
        if not text:
            return []

        if not self.special_tokens_pattern:
            # 无 special，直接全量走 Trie
            data = text.encode("utf-8", errors="strict")
            return self._encode_bytes_with_trie(data)

        ids: List[int] = []
        pos = 0
        for m in self.special_tokens_pattern.finditer(text):
            start, end = m.span()
            # 先处理 special 前面的普通片段
            if start > pos:
                chunk = text[pos:start].encode("utf-8", errors="strict")
                ids.extend(self._encode_bytes_with_trie(chunk))

            # 处理 special 本身
            ssp = m.group(0)
            tid = self.special_to_id.get(ssp)
            if tid is None:
                # 若未声明为 special（或不在 vocab），回退成普通文本处理
                # 一般不会触发，因为 __init__ 已校验 special 均在 vocab
                ids.extend(self._encode_bytes_with_trie(ssp.encode("utf-8", errors="strict")))
            else:
                ids.append(tid)

            pos = end

        # 处理最后的尾部普通片段
        if pos < len(text):
            tail = text[pos:].encode("utf-8", errors="strict")
            ids.extend(self._encode_bytes_with_trie(tail))

        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码：对一个字符串可迭代对象逐个编码并按顺序产出 token id。
        - 与 encode(text) 的切分逻辑一致（包含 special_tokens 处理）
        - 不会在元素之间插入任何分隔符
        - 不跨元素匹配 special token
        """
        if self.special_tokens_pattern is None:
            # 无 special，直接按块进行字节级最长匹配
            for text in iterable:
                if not text:
                    continue
                data = text.encode("utf-8", errors="strict")
                for tid in self._encode_bytes_with_trie(data):
                    yield tid
            return

        # 有 special，逐块在字符层面先切分 special，再进行字节级匹配
        for text in iterable:
            if not text:
                continue

            pos = 0
            for m in self.special_tokens_pattern.finditer(text):
                start, end = m.span()
                # 先处理 special 前面的普通片段
                if start > pos:
                    chunk = text[pos:start].encode("utf-8", errors="strict")
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
            if pos < len(text):
                tail = text[pos:].encode("utf-8", errors="strict")
                for tid in self._encode_bytes_with_trie(tail):
                    yield tid

    def decode(self, ids: List[int]) -> str:
        """
        将 token id 列表还原为字符串。
        - 先拼接 bytes，再用 UTF-8 解码；
        - 使用 errors="replace" 以容忍任何非标准字节序列，避免解码异常。
        """
        if not ids:
            return ""
        try:
            data = b"".join(self.id_to_bytes[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"Unknown token id in sequence: {e!r}")

        # 容错解码：遇到非法 UTF-8 序列，用 U+FFFD 替换
        return data.decode("utf-8", errors="replace")