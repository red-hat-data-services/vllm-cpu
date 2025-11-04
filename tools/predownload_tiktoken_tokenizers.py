# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""helper utility to use tiktoken to pre-download tokenizer files

This is useful for disconnected environments in which internet
access is not available.

For more details:
- https://github.com/openai/tiktoken/blob/0.12.0/tiktoken/load.py?plain=1#L38-L41
- https://issues.redhat.com/browse/INFERENG-2959
"""

import tiktoken
import os

TIKTOKEN_CACHE_DIRS = (
    "TIKTOKEN_CACHE_DIR",
    "DATA_GYM_CACHE_DIR",
)
cache_dir = next(
    (
        os.environ.get(cache_dir_env)
        for cache_dir_env in TIKTOKEN_CACHE_DIRS
        if cache_dir_env in os.environ
    ),
    None,
)
if cache_dir is None:
    cache_dir = "/tmp/tiktoken_cache_dir"
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    print(f"Set TIKTOKEN_CACHE_DIR={cache_dir}")

to_download = (
    "cl100k_base",
    "o200k_base",
)

print(f"Downloading tokenizers to {cache_dir=}. Tokenizer list: {to_download}")
os.makedirs(cache_dir, exist_ok=True)
for tokenizer in to_download:
    # downloads the tokenizer to the cache_directory
    _ = tiktoken.get_encoding(tokenizer)

print(
    "Retrieved tokenizers. Contents of cache_dir (sha1sum of tokenizer URL): "
    + ", ".join(os.listdir(cache_dir)),
)
