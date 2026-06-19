"""
Efficiency Metrics
Tracks performance and resource usage
"""

import time
import psutil
import os
from typing import Dict, Any, List
from collections import defaultdict
import pandas as pd


class EfficiencyMetrics:
    """Track efficiency metrics for RAG pipeline"""
    
    def __init__(self):
        """Initialize efficiency metrics tracker"""
        self.metrics = defaultdict(list)
        self.timers = {}
    
    def start_timer(self, operation_name: str):
        """
        Start a timer for an operation
        
        Args:
            operation_name: Name of the operation to time
        """
        self.timers[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """
        End a timer and record the elapsed time
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Elapsed time in seconds
        """
        if operation_name not in self.timers:
            raise ValueError(f"Timer '{operation_name}' was not started")
        
        elapsed = time.time() - self.timers[operation_name]
        self.metrics[f'{operation_name}_time'].append(elapsed)
        del self.timers[operation_name]
        
        return elapsed
    
    def record_metric(self, metric_name: str, value: float):
        """
        Record a metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name].append(value)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage
        
        Returns:
            Dictionary with memory metrics in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        }
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage
        
        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent(interval=0.1)
    
    def record_embedding_metrics(self,
                                 num_chunks: int,
                                 embedding_time: float,
                                 embedding_dim: int):
        """
        Record embedding-related metrics
        
        Args:
            num_chunks: Number of chunks embedded
            embedding_time: Time taken for embedding
            embedding_dim: Embedding dimension
        """
        self.record_metric('num_chunks', num_chunks)
        self.record_metric('embedding_time', embedding_time)
        self.record_metric('embedding_time_per_chunk', embedding_time / num_chunks if num_chunks > 0 else 0)
        self.record_metric('embedding_dim', embedding_dim)
    
    def record_index_metrics(self,
                            index_build_time: float,
                            index_size_mb: float,
                            num_vectors: int):
        """
        Record vector index metrics
        
        Args:
            index_build_time: Time to build index
            index_size_mb: Size of index in MB
            num_vectors: Number of vectors in index
        """
        self.record_metric('index_build_time', index_build_time)
        self.record_metric('index_size_mb', index_size_mb)
        self.record_metric('num_vectors', num_vectors)
    
    def record_retrieval_metrics(self,
                                query_latency: float,
                                num_retrieved: int):
        """
        Record retrieval metrics
        
        Args:
            query_latency: Time for retrieval query
            num_retrieved: Number of chunks retrieved
        """
        self.record_metric('query_latency', query_latency)
        self.record_metric('num_retrieved', num_retrieved)
    
    def record_generation_metrics(self,
                                  generation_time: float,
                                  tokens_generated: int,
                                  tokens_per_second: float):
        """
        Record generation metrics
        
        Args:
            generation_time: Time for generation
            tokens_generated: Number of tokens generated
            tokens_per_second: Generation speed
        """
        self.record_metric('generation_time', generation_time)
        self.record_metric('tokens_generated', tokens_generated)
        self.record_metric('tokens_per_second', tokens_per_second)

    
    def record_end_to_end_latency(self, latency: float):
        """
        Record end-to-end query latency
        
        Args:
            latency: Total latency from query to answer
        """
        self.record_metric('end_to_end_latency', latency)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics
        
        Returns:
            Dictionary of statistics for each metric
        """
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                summary[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': self._std(values) if len(values) > 1 else 0.0,
                    'count': len(values)
                }
        
        return summary
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to pandas DataFrame
        
        Returns:
            DataFrame with all metrics
        """
        # Find maximum length
        max_len = max(len(v) for v in self.metrics.values()) if self.metrics else 0
        
        # Pad shorter lists with None
        padded_metrics = {}
        for key, values in self.metrics.items():
            padded = values + [None] * (max_len - len(values))
            padded_metrics[key] = padded
        
        return pd.DataFrame(padded_metrics)
    
    def save_to_csv(self, filepath: str):
        """
        Save metrics to CSV file
        
        Args:
            filepath: Path to save CSV
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to {filepath}")
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = defaultdict(list)
        self.timers = {}
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("EFFICIENCY METRICS SUMMARY")
        print("="*60)
        
        for metric_name, stats in summary.items():
            print(f"\n{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Count: {stats['count']}")


class StrategyComparison:
    """Compare efficiency across different chunking strategies"""
    
    def __init__(self):
        """Initialize strategy comparison"""
        self.strategy_metrics = {}
    
    def add_strategy_metrics(self, strategy_name: str, metrics: EfficiencyMetrics):
        """
        Add metrics for a chunking strategy
        
        Args:
            strategy_name: Name of the strategy
            metrics: EfficiencyMetrics instance
        """
        self.strategy_metrics[strategy_name] = metrics.get_summary_statistics()
    
    def compare_strategies(self, metric_name: str) -> Dict[str, float]:
        """
        Compare a specific metric across strategies
        
        Args:
            metric_name: Name of metric to compare
            
        Returns:
            Dictionary mapping strategy names to metric values
        """
        comparison = {}
        
        for strategy, metrics in self.strategy_metrics.items():
            if metric_name in metrics:
                comparison[strategy] = metrics[metric_name]['mean']
        
        return comparison
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate comparison report as DataFrame
        
        Returns:
            DataFrame comparing all strategies
        """
        # Collect all metric names
        all_metrics = set()
        for metrics in self.strategy_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Build comparison data
        data = []
        for strategy, metrics in self.strategy_metrics.items():
            row = {'strategy': strategy}
            for metric_name in all_metrics:
                if metric_name in metrics:
                    row[f'{metric_name}_mean'] = metrics[metric_name]['mean']
                    row[f'{metric_name}_std'] = metrics[metric_name]['std']
                else:
                    row[f'{metric_name}_mean'] = None
                    row[f'{metric_name}_std'] = None
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_comparison(self, filepath: str):
        """
        Save comparison report to CSV
        
        Args:
            filepath: Path to save CSV
        """
        df = self.generate_comparison_report()
        df.to_csv(filepath, index=False)
        print(f"Comparison saved to {filepath}")


if __name__ == "__main__":
    # Test efficiency metrics
    metrics = EfficiencyMetrics()
    
    # Simulate embedding
    metrics.start_timer('embedding')
    time.sleep(0.1)  # Simulate work
    metrics.end_timer('embedding')
    metrics.record_embedding_metrics(100, 0.1, 384)
    
    # Simulate retrieval
    metrics.record_retrieval_metrics(0.05, 5)
    
    
    # Print summary
    metrics.print_summary()
    
    # Get memory usage
    memory = metrics.get_memory_usage()
    print(f"\nMemory Usage: {memory['rss_mb']:.2f} MB")
