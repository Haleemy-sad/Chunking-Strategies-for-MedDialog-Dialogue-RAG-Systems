"""
Retrieval Evaluation Metrics - Aligned with MedRAG and Base Papers

Implements comprehensive retrieval metrics as used in:
- MedRAG (2024): Recall@K, Precision@K, MRR, Hit Rate
- Self-RAG (2023): F1, EM (Exact Match)
- Standard IR metrics: nDCG@K, MAP

References:
- Xiong et al. (2024). "Benchmarking Retrieval-Augmented Generation for Medicine"
- Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique"
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


class RetrievalMetrics:
    """
    Calculate retrieval quality metrics aligned with base papers
    
    Implements metrics from MedRAG, Self-RAG, and standard IR literature
    """
    
    def __init__(self):
        """Initialize retrieval metrics calculator"""
        self.metrics_history = defaultdict(list)
    
    def recall_at_k(self, 
                   retrieved_ids: List[str],
                   relevant_ids: Set[str],
                   k: int = None) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            k: Consider only top K results (None = all)
            
        Returns:
            Recall@K score (0 to 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        if len(relevant_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_ids)
        
        recall = num_relevant_retrieved / len(relevant_ids)
        return recall
    
    def mean_reciprocal_rank(self,
                            retrieved_ids: List[str],
                            relevant_ids: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            
        Returns:
            MRR score (0 to 1)
        """
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                return 1.0 / rank
        
        return 0.0
    
    def ndcg_at_k(self,
                 retrieved_ids: List[str],
                 relevance_scores: Dict[str, float],
                 k: int = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@K)
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevance_scores: Dictionary mapping chunk IDs to relevance scores
            k: Consider only top K results (None = all)
            
        Returns:
            nDCG@K score (0 to 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            relevance = relevance_scores.get(chunk_id, 0.0)
            dcg += (2 ** relevance - 1) / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)
        if k is not None:
            sorted_relevance = sorted_relevance[:k]
        
        idcg = 0.0
        for rank, relevance in enumerate(sorted_relevance, start=1):
            idcg += (2 ** relevance - 1) / np.log2(rank + 1)
        
        # Calculate nDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def precision_at_k(self,
                      retrieved_ids: List[str],
                      relevant_ids: Set[str],
                      k: int = None) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            k: Consider only top K results (None = all)
            
        Returns:
            Precision@K score (0 to 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        if len(retrieved_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_ids)
        
        precision = num_relevant_retrieved / len(retrieved_ids)
        return precision
    
    def f1_score(self,
                retrieved_ids: List[str],
                relevant_ids: Set[str],
                k: int = None) -> float:
        """
        Calculate F1 score
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            k: Consider only top K results (None = all)
            
        Returns:
            F1 score (0 to 1)
        """
        precision = self.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = self.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def hit_rate_at_k(self,
                     retrieved_ids: List[str],
                     relevant_ids: Set[str],
                     k: int = None) -> float:
        """
        Calculate Hit Rate@K (Success@K)
        Used in MedRAG (2024)
        
        1 if at least one relevant document in top-K, 0 otherwise
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs
            k: Consider only top K results
            
        Returns:
            Hit rate (0 or 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        retrieved_set = set(retrieved_ids)
        has_relevant = len(retrieved_set & relevant_ids) > 0
        
        return 1.0 if has_relevant else 0.0
    
    def average_precision(self,
                         retrieved_ids: List[str],
                         relevant_ids: Set[str]) -> float:
        """
        Calculate Average Precision (AP)
        Used for MAP (Mean Average Precision)
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs
            
        Returns:
            Average Precision score (0 to 1)
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        num_relevant = 0
        precision_sum = 0.0
        
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                precision_sum += precision_at_rank
        
        if num_relevant == 0:
            return 0.0
        
        ap = precision_sum / len(relevant_ids)
        return ap
    
    def mean_average_precision(self, all_results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries
        
        Args:
            all_results: List of (retrieved_ids, relevant_ids) tuples
            
        Returns:
            MAP score (0 to 1)
        """
        if len(all_results) == 0:
            return 0.0
        
        ap_scores = [self.average_precision(retrieved, relevant) 
                    for retrieved, relevant in all_results]
        
        return np.mean(ap_scores)
    
    def evaluate_retrieval(self,
                          retrieved_chunks: List[Dict[str, Any]],
                          relevant_chunk_ids: Set[str],
                          relevance_scores: Dict[str, float] = None,
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate retrieval with multiple metrics (aligned with MedRAG 2024)
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            relevant_chunk_ids: Set of relevant chunk IDs
            relevance_scores: Optional relevance scores for nDCG
            k_values: List of K values to evaluate (default: [1, 3, 5, 10])
            
        Returns:
            Dictionary of metrics matching base paper format
        """
        retrieved_ids = [chunk.get('chunk_id', chunk.get('id', '')) for chunk in retrieved_chunks]
        
        metrics = {
            'mrr': self.mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids),
            'map': self.average_precision(retrieved_ids, relevant_chunk_ids)
        }
        
        # Metrics at different K values (as in MedRAG)
        for k in k_values:
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved_ids, relevant_chunk_ids, k)
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved_ids, relevant_chunk_ids, k)
            metrics[f'f1@{k}'] = self.f1_score(retrieved_ids, relevant_chunk_ids, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(retrieved_ids, relevant_chunk_ids, k)
            
            if relevance_scores is not None:
                metrics[f'ndcg@{k}'] = self.ndcg_at_k(retrieved_ids, relevance_scores, k)
        
        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        return metrics
    
    def evaluate_multiple_queries(self,
                                  all_retrievals: List[Tuple[List[Dict], Set[str]]],
                                  k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval performance across multiple queries
        Returns averaged metrics - format matching MedRAG paper
        
        Args:
            all_retrievals: List of (retrieved_chunks, relevant_ids) tuples
            k_values: K values to evaluate
            
        Returns:
            Dictionary of averaged metrics
        """
        self.reset_history()
        
        for retrieved_chunks, relevant_ids in all_retrievals:
            self.evaluate_retrieval(retrieved_chunks, relevant_ids, k_values=k_values)
        
        avg_metrics = self.get_average_metrics()
        return avg_metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of all metrics in history"""
        avg_metrics = {}
        for key, values in self.metrics_history.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
        return avg_metrics
    
    def reset_history(self):
        """Reset metrics history"""
        self.metrics_history = defaultdict(list)


if __name__ == "__main__":
    # Test retrieval metrics
    metrics = RetrievalMetrics()
    
    # Example retrieval
    retrieved_ids = ['chunk_1', 'chunk_3', 'chunk_7', 'chunk_2', 'chunk_9']
    relevant_ids = {'chunk_1', 'chunk_2', 'chunk_5'}
    relevance_scores = {
        'chunk_1': 1.0,
        'chunk_2': 1.0,
        'chunk_3': 0.0,
        'chunk_5': 1.0,
        'chunk_7': 0.0,
        'chunk_9': 0.0
    }
    
    # Create chunk dictionaries
    retrieved_chunks = [{'chunk_id': cid, 'text': f'Text for {cid}'} for cid in retrieved_ids]
    
    # Evaluate
    results = metrics.evaluate_retrieval(
        retrieved_chunks,
        relevant_ids,
        relevance_scores,
        k_values=[3, 5]
    )
    
    print("Retrieval Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
"""
Retrieval Evaluation Metrics - Aligned with MedRAG and Base Papers

Implements comprehensive retrieval metrics as used in:
- MedRAG (2024): Recall@K, Precision@K, MRR, Hit Rate
- Self-RAG (2023): F1, EM (Exact Match)
- Standard IR metrics: nDCG@K, MAP

References:
- Xiong et al. (2024). "Benchmarking Retrieval-Augmented Generation for Medicine"
- Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique"
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


class RetrievalMetrics:
    """
    Calculate retrieval quality metrics aligned with base papers
    
    Implements metrics from MedRAG, Self-RAG, and standard IR literature
    """
    
    def __init__(self):
        """Initialize retrieval metrics calculator"""
        self.metrics_history = defaultdict(list)
    
    def recall_at_k(self, 
                   retrieved_ids: List[str],
                   relevant_ids: Set[str],
                   k: int = None) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            k: Consider only top K results (None = all)
            
        Returns:
            Recall@K score (0 to 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        if len(relevant_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_ids)
        
        recall = num_relevant_retrieved / len(relevant_ids)
        return recall
    
    def mean_reciprocal_rank(self,
                            retrieved_ids: List[str],
                            relevant_ids: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            
        Returns:
            MRR score (0 to 1)
        """
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                return 1.0 / rank
        
        return 0.0
    
    def ndcg_at_k(self,
                 retrieved_ids: List[str],
                 relevance_scores: Dict[str, float],
                 k: int = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@K)
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevance_scores: Dictionary mapping chunk IDs to relevance scores
            k: Consider only top K results (None = all)
            
        Returns:
            nDCG@K score (0 to 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            relevance = relevance_scores.get(chunk_id, 0.0)
            dcg += (2 ** relevance - 1) / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)
        if k is not None:
            sorted_relevance = sorted_relevance[:k]
        
        idcg = 0.0
        for rank, relevance in enumerate(sorted_relevance, start=1):
            idcg += (2 ** relevance - 1) / np.log2(rank + 1)
        
        # Calculate nDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def precision_at_k(self,
                      retrieved_ids: List[str],
                      relevant_ids: Set[str],
                      k: int = None) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            k: Consider only top K results (None = all)
            
        Returns:
            Precision@K score (0 to 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        if len(retrieved_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_ids)
        
        precision = num_relevant_retrieved / len(retrieved_ids)
        return precision
    
    def f1_score(self,
                retrieved_ids: List[str],
                relevant_ids: Set[str],
                k: int = None) -> float:
        """
        Calculate F1 score
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs (ground truth)
            k: Consider only top K results (None = all)
            
        Returns:
            F1 score (0 to 1)
        """
        precision = self.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = self.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def hit_rate_at_k(self,
                     retrieved_ids: List[str],
                     relevant_ids: Set[str],
                     k: int = None) -> float:
        """
        Calculate Hit Rate@K (Success@K)
        Used in MedRAG (2024)
        
        1 if at least one relevant document in top-K, 0 otherwise
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs
            k: Consider only top K results
            
        Returns:
            Hit rate (0 or 1)
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]
        
        retrieved_set = set(retrieved_ids)
        has_relevant = len(retrieved_set & relevant_ids) > 0
        
        return 1.0 if has_relevant else 0.0
    
    def average_precision(self,
                         retrieved_ids: List[str],
                         relevant_ids: Set[str]) -> float:
        """
        Calculate Average Precision (AP)
        Used for MAP (Mean Average Precision)
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs
            
        Returns:
            Average Precision score (0 to 1)
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        num_relevant = 0
        precision_sum = 0.0
        
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                precision_sum += precision_at_rank
        
        if num_relevant == 0:
            return 0.0
        
        ap = precision_sum / len(relevant_ids)
        return ap
    
    def mean_average_precision(self, all_results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries
        
        Args:
            all_results: List of (retrieved_ids, relevant_ids) tuples
            
        Returns:
            MAP score (0 to 1)
        """
        if len(all_results) == 0:
            return 0.0
        
        ap_scores = [self.average_precision(retrieved, relevant) 
                    for retrieved, relevant in all_results]
        
        return np.mean(ap_scores)
    
    def evaluate_retrieval(self,
                          retrieved_chunks: List[Dict[str, Any]],
                          relevant_chunk_ids: Set[str],
                          relevance_scores: Dict[str, float] = None,
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate retrieval with multiple metrics (aligned with MedRAG 2024)
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            relevant_chunk_ids: Set of relevant chunk IDs
            relevance_scores: Optional relevance scores for nDCG
            k_values: List of K values to evaluate (default: [1, 3, 5, 10])
            
        Returns:
            Dictionary of metrics matching base paper format
        """
        retrieved_ids = [chunk.get('chunk_id', chunk.get('id', '')) for chunk in retrieved_chunks]
        
        metrics = {
            'mrr': self.mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids),
            'map': self.average_precision(retrieved_ids, relevant_chunk_ids)
        }
        
        # Metrics at different K values (as in MedRAG)
        for k in k_values:
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved_ids, relevant_chunk_ids, k)
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved_ids, relevant_chunk_ids, k)
            metrics[f'f1@{k}'] = self.f1_score(retrieved_ids, relevant_chunk_ids, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(retrieved_ids, relevant_chunk_ids, k)
            
            if relevance_scores is not None:
                metrics[f'ndcg@{k}'] = self.ndcg_at_k(retrieved_ids, relevance_scores, k)
        
        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        return metrics
    
    def evaluate_multiple_queries(self,
                                  all_retrievals: List[Tuple[List[Dict], Set[str]]],
                                  k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval performance across multiple queries
        Returns averaged metrics - format matching MedRAG paper
        
        Args:
            all_retrievals: List of (retrieved_chunks, relevant_ids) tuples
            k_values: K values to evaluate
            
        Returns:
            Dictionary of averaged metrics
        """
        self.reset_history()
        
        for retrieved_chunks, relevant_ids in all_retrievals:
            self.evaluate_retrieval(retrieved_chunks, relevant_ids, k_values=k_values)
        
        avg_metrics = self.get_average_metrics()
        return avg_metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of all metrics in history"""
        avg_metrics = {}
        for key, values in self.metrics_history.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
        return avg_metrics
    
    def reset_history(self):
        """Reset metrics history"""
        self.metrics_history = defaultdict(list)


if __name__ == "__main__":
    # Test retrieval metrics
    metrics = RetrievalMetrics()
    
    # Example retrieval
    retrieved_ids = ['chunk_1', 'chunk_3', 'chunk_7', 'chunk_2', 'chunk_9']
    relevant_ids = {'chunk_1', 'chunk_2', 'chunk_5'}
    relevance_scores = {
        'chunk_1': 1.0,
        'chunk_2': 1.0,
        'chunk_3': 0.0,
        'chunk_5': 1.0,
        'chunk_7': 0.0,
        'chunk_9': 0.0
    }
    
    # Create chunk dictionaries
    retrieved_chunks = [{'chunk_id': cid, 'text': f'Text for {cid}'} for cid in retrieved_ids]
    
    # Evaluate
    results = metrics.evaluate_retrieval(
        retrieved_chunks,
        relevant_ids,
        relevance_scores,
        k_values=[3, 5]
    )
    
    print("Retrieval Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
