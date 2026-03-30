"""Module-level LoRA file cache.

Caches raw LoRA state dicts after disk load (and optional QKV conversion)
so that repeated loads of the same file across batch images, slots, and
chained PersonDetailer nodes hit memory instead of disk.
"""

from collections import OrderedDict

import comfy.utils


class _LoraFileCache:
    """LRU cache for loaded LoRA state dicts, shared across all node instances."""

    _cache = OrderedDict()  # lora_path -> state_dict
    _max_entries = 6

    @classmethod
    def get(cls, lora_path, safe_load=True):
        if lora_path in cls._cache:
            cls._cache.move_to_end(lora_path)
            print(f"[FVMTools] LoRA cache hit: {lora_path.split('/')[-1].split(chr(92))[-1]}")
            return cls._cache[lora_path]
        sd = comfy.utils.load_torch_file(lora_path, safe_load=safe_load)
        if len(cls._cache) >= cls._max_entries:
            evicted_key, _ = cls._cache.popitem(last=False)
            evicted_name = evicted_key.split("/")[-1].split("\\")[-1]
            print(f"[FVMTools] LoRA cache evict: {evicted_name}")
        cls._cache[lora_path] = sd
        return sd

    @classmethod
    def clear(cls):
        cls._cache.clear()
