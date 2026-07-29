"""
搜索缓存模块 — 本地知识库缓存，避免重复调用检索服务器。

用法:
    cache = SearchCache(cache_dir="search_cache")
    
    # 首次调用 -> 走 HTTP 请求
    result = cache.get_or_query("什么是机器学习", query_func=your_query_function)
    
    # 相同 query -> 直接从本地缓存返回
    result = cache.get_or_query("什么是机器学习", query_func=your_query_function)
"""

import os
import json
import hashlib
import time
import threading
from typing import Callable, Optional


class SearchCache:
    """线程安全的搜索缓存，支持内存 + 磁盘两级缓存。"""

    def __init__(self, cache_dir: str = None, use_memory: bool = True):
        """
        Args:
            cache_dir: 磁盘缓存目录。为 None 则不持久化到磁盘。
            use_memory: 是否使用内存缓存（速度更快）。
        """
        self._memory_cache = {} if use_memory else None
        self._cache_dir = cache_dir
        self._lock = threading.Lock()  # 线程安全

        # 统计信息
        self._hits = 0
        self._misses = 0

        # 加载已有磁盘缓存
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_from_disk()

    def _hash_key(self, query: str) -> str:
        """用 MD5 对 query 做哈希，避免文件名非法字符。"""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _cache_path(self, query: str) -> str:
        """返回磁盘缓存路径。"""
        h = self._hash_key(query)
        return os.path.join(self._cache_dir, f"{h}.json")

    def _load_from_disk(self):
        """扫描磁盘缓存目录，载入内存。"""
        if self._cache_dir is None or not os.path.exists(self._cache_dir):
            return
        loaded = 0
        for fname in os.listdir(self._cache_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self._cache_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    query = data.get("query")
                    result = data.get("result")
                    if query is not None and result is not None:
                        if self._memory_cache is not None:
                            self._memory_cache[query] = result
                        loaded += 1
                except Exception:
                    pass
        if loaded > 0:
            print(f"[SearchCache] Loaded {loaded} cached entries from {self._cache_dir}")

    def get(self, query: str) -> Optional[list]:
        """尝试从缓存获取结果。"""
        if self._memory_cache is not None:
            with self._lock:
                result = self._memory_cache.get(query)
            if result is not None:
                self._hits += 1
                return result

        # 尝试磁盘缓存
        if self._cache_dir is not None:
            fpath = self._cache_path(query)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    result = data.get("result")
                    if result is not None:
                        # 同步到内存
                        if self._memory_cache is not None:
                            with self._lock:
                                self._memory_cache[query] = result
                        self._hits += 1
                        return result
                except Exception:
                    pass

        self._misses += 1
        return None

    def set(self, query: str, result) -> None:
        """写入缓存。"""
        if self._memory_cache is not None:
            with self._lock:
                self._memory_cache[query] = result

        if self._cache_dir is not None:
            fpath = self._cache_path(query)
            try:
                with open(fpath, "w") as f:
                    json.dump({"query": query, "result": result, "timestamp": time.time()}, f)
            except Exception as e:
                print(f"[SearchCache] Warning: failed to write cache: {e}")

    def get_or_query(self, query: str, query_func: Callable, *args, **kwargs):
        """缓存优先，未命中则调用 query_func 获取并缓存。

        Args:
            query: 搜索查询字符串（用作缓存 key）。
            query_func: 实际执行搜索的函数。应是 callable(query) -> result。
            *args, **kwargs: 透传给 query_func 的额外参数。

        Returns:
            搜索结果（格式由 query_func 决定）。
        """
        cached = self.get(query)
        if cached is not None:
            return cached

        result = query_func(query, *args, **kwargs)
        self.set(query, result)
        return result

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
        }

    def print_stats(self):
        s = self.stats
        total = s["hits"] + s["misses"]
        print(f"[SearchCache] Hits: {s['hits']}, Misses: {s['misses']}, "
              f"Hit Rate: {s['hit_rate']:.1%}, Total: {total}")

    def __len__(self):
        if self._memory_cache is not None:
            return len(self._memory_cache)
        if self._cache_dir is not None and os.path.exists(self._cache_dir):
            return len([f for f in os.listdir(self._cache_dir) if f.endswith(".json")])
        return 0
