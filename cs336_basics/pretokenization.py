import os
from typing import BinaryIO, List
import collections
import regex
from multiprocessing import Pool
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

# Pre-compute cache for byte-to-bytes conversion to optimize hot path
BYTE_CACHE = {i: bytes([i]) for i in range(256)}

PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def pretokenize( 
        input_path: str | os.PathLike,
        special_tokens: list[str],
        **kwargs) -> collections.Counter[tuple[bytes]]:
    pretoken_byte_counts = collections.Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            kwargs.get("num_processes", 1),
            [t.encode("utf-8") for t in special_tokens],
        )

        chunk_args = [
            ProcessChunkArgs(start, end, input_path, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with Pool(kwargs.get("num_processes", 1)) as pool:
            for chunk_pretoken_byte_counts in pool.imap_unordered(
                process_chunk, chunk_args
            ):
                pretoken_byte_counts.update(chunk_pretoken_byte_counts)

    logger.info("Done counting all pretoken bytes")

    return pretoken_byte_counts

def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    special_tokens_pattern = b"|".join(regex.escape(t) for t in special_tokens)
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            match = regex.search(special_tokens_pattern, mini_chunk)
            if match:
                chunk_boundaries[bi] = initial_position + match.end()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

@dataclass
class ProcessChunkArgs:
    start: int
    end: int
    input_path: str | os.PathLike
    special_tokens: list[str]

def process_chunk(args: ProcessChunkArgs) -> collections.Counter[tuple[bytes]]:
    with open(args.input_path, "rb") as f:
        f.seek(args.start)
        chunk = f.read(args.end - args.start).decode("utf-8", errors="ignore")
    logger.info(f"Counting pretoken bytes for {(args.start, args.end)}")
    if args.special_tokens:
        special_tokens_pattern = "|".join(regex.escape(t) for t in sorted(args.special_tokens, key=len, reverse=True))
        splits = regex.split(special_tokens_pattern, chunk)
    else:
        splits = [chunk]

    pretoken_byte_counts = collections.Counter(
        tuple(BYTE_CACHE[b] for b in pretoken.encode("utf-8"))
        for split in splits
        for pretoken in regex.findall(PRETOKEN_PATTERN, split)
    )
    logger.info(f"Done counting pretoken bytes for {(args.start, args.end)}")
    return pretoken_byte_counts

