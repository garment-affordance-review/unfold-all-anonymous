"""Performance profiling utilities for identifying bottlenecks."""

import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List
import torch


class PerformanceProfiler:
    """Lightweight profiler for tracking execution time of code sections."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self._stack: List[tuple] = []  # (name, start_time)
        
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code section.
        
        Usage:
            with profiler.profile("section_name"):
                # code to profile
                pass
        """
        if not self.enabled:
            yield
            return
            
        # Synchronize CUDA before starting
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        self._stack.append((name, start_time))
        
        try:
            yield
        finally:
            # Synchronize CUDA before measuring
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            self._stack.pop()
            self.timings[name].append(elapsed)
            self.call_counts[name] += 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a profiled section."""
        if name not in self.timings or not self.timings[name]:
            return {}
            
        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }
    
    def print_stats(self, min_time_ms: float = 0.1):
        """Print profiling statistics.
        
        Args:
            min_time_ms: Only show sections taking more than this many milliseconds on average
        """
        if not self.enabled or not self.timings:
            return
            
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING RESULTS")
        print("="*80)
        
        # Sort by total time descending
        items = []
        for name in self.timings:
            stats = self.get_stats(name)
            if stats['mean'] * 1000 >= min_time_ms:
                items.append((name, stats))
        
        items.sort(key=lambda x: x[1]['total'], reverse=True)
        
        print(f"{'Section':<50} {'Calls':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("-"*80)
        
        for name, stats in items:
            print(f"{name:<50} {stats['count']:>8} {stats['total']:>10.3f} "
                  f"{stats['mean']*1000:>10.2f} {stats['min']*1000:>10.2f} {stats['max']*1000:>10.2f}")
        
        print("="*80 + "\n")
    
    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        self._stack.clear()


# Global profiler instance
_global_profiler = PerformanceProfiler(enabled=False)


def enable_profiling():
    """Enable global profiling."""
    global _global_profiler
    _global_profiler.enabled = True


def disable_profiling():
    """Disable global profiling."""
    global _global_profiler
    _global_profiler.enabled = False


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


@contextmanager
def profile(name: str):
    """Convenience function for profiling using global profiler."""
    with _global_profiler.profile(name):
        yield
