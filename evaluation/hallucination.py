"""
Hallucination Detection
Classifies generated answers into hallucination types
"""

import re
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer


class HallucinationDetector:
    """Detect and classify hallucinations in generated text"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize hallucination detector
        
        Args:
            embedding_model_name: Name of sentence transformer model
        """
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Medical harm keywords
        self.harm_keywords = [
            'deadly', 'fatal', 'lethal', 'dangerous', 'severe',
            'emergency', 'critical', 'life-threatening',
            'immediately', 'urgent', 'serious'
        ]
        
        # Confidence indicators
        self.uncertain_phrases = [
            'i think', 'maybe', 'perhaps', 'possibly', 'might',
            'could be', 'may be', 'probably', 'likely',
            'i don\'t know', 'not sure', 'unclear'
        ]
    
    def detect_factual_hallucination(self,
                                    generated_answer: str,
                                    context_chunks: List[Dict[str, Any]],
                                    similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Detect factual hallucination by comparing answer to context
        
        Args:
            generated_answer: Generated answer text
            context_chunks: Retrieved context chunks
            similarity_threshold: Threshold for considering text as supported
            
        Returns:
            Dictionary with hallucination detection results
        """
        # Combine context
        context_text = " ".join([chunk['text'] for chunk in context_chunks])
        
        # Embed answer and context
        answer_emb = self.embedder.encode(generated_answer, convert_to_tensor=False)
        context_emb = self.embedder.encode(context_text, convert_to_tensor=False)
        
        # Calculate semantic similarity
        similarity = np.dot(answer_emb, context_emb) / (
            np.linalg.norm(answer_emb) * np.linalg.norm(context_emb)
        )
        
        # Check if answer is grounded in context
        is_hallucinated = similarity < similarity_threshold
        
        # Check if answer explicitly states uncertainty
        is_uncertain = any(phrase in generated_answer.lower() for phrase in self.uncertain_phrases)
        
        return {
            'is_factual_hallucination': is_hallucinated and not is_uncertain,
            'semantic_similarity': float(similarity),
            'is_uncertain': is_uncertain,
            'threshold': similarity_threshold
        }
    
    def detect_reasoning_hallucination(self,
                                      generated_answer: str,
                                      context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect reasoning hallucination (invalid logical inference)
        
        Args:
            generated_answer: Generated answer text
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary with reasoning hallucination detection
        """
        # Check for causal/reasoning phrases
        reasoning_phrases = [
            'therefore', 'thus', 'hence', 'consequently',
            'this means', 'this suggests', 'this indicates',
            'because', 'since', 'as a result'
        ]
        
        has_reasoning = any(phrase in generated_answer.lower() for phrase in reasoning_phrases)
        
        # Check if reasoning is present in context
        context_text = " ".join([chunk['text'] for chunk in context_chunks]).lower()
        reasoning_in_context = any(phrase in context_text for phrase in reasoning_phrases)
        
        # If answer contains reasoning not present in context, flag as potential hallucination
        is_reasoning_hallucination = has_reasoning and not reasoning_in_context
        
        return {
            'is_reasoning_hallucination': is_reasoning_hallucination,
            'has_reasoning': has_reasoning,
            'reasoning_in_context': reasoning_in_context
        }
    
    def assess_medical_harm_risk(self, generated_answer: str) -> Dict[str, Any]:
        """
        Assess risk of medical harm from generated answer
        
        Args:
            generated_answer: Generated answer text
            
        Returns:
            Dictionary with harm risk assessment
        """
        answer_lower = generated_answer.lower()
        
        # Count harm keywords
        harm_mentions = sum(1 for keyword in self.harm_keywords if keyword in answer_lower)
        
        # Check for medical advice without proper disclaimer
        advice_phrases = [
            'you should', 'you must', 'take', 'use',
            'do not', 'avoid', 'stop'
        ]
        
        gives_advice = any(phrase in answer_lower for phrase in advice_phrases)
        
        # Check for disclaimer
        disclaimer_phrases = [
            'consult', 'see a doctor', 'medical professional',
            'healthcare provider', 'seek medical attention'
        ]
        
        has_disclaimer = any(phrase in answer_lower for phrase in disclaimer_phrases)
        
        # Assess risk level
        if harm_mentions >= 3 and gives_advice and not has_disclaimer:
            risk_level = 'high'
        elif harm_mentions >= 2 or (gives_advice and not has_disclaimer):
            risk_level = 'medium'
        elif harm_mentions >= 1 or gives_advice:
            risk_level = 'low'
        else:
            risk_level = 'none'
        
        return {
            'medical_harm_risk': risk_level,
            'harm_keyword_count': harm_mentions,
            'gives_medical_advice': gives_advice,
            'has_disclaimer': has_disclaimer
        }
    
    def comprehensive_hallucination_check(self,
                                         generated_answer: str,
                                         context_chunks: List[Dict[str, Any]],
                                         similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Perform comprehensive hallucination detection
        
        Args:
            generated_answer: Generated answer text
            context_chunks: Retrieved context chunks
            similarity_threshold: Threshold for factual hallucination
            
        Returns:
            Dictionary with all hallucination metrics
        """
        # Factual hallucination
        factual = self.detect_factual_hallucination(
            generated_answer,
            context_chunks,
            similarity_threshold
        )
        
        # Reasoning hallucination
        reasoning = self.detect_reasoning_hallucination(
            generated_answer,
            context_chunks
        )
        
        # Medical harm risk
        harm = self.assess_medical_harm_risk(generated_answer)
        
        # Overall hallucination flag
        has_hallucination = (
            factual['is_factual_hallucination'] or
            reasoning['is_reasoning_hallucination']
        )
        
        return {
            **factual,
            **reasoning,
            **harm,
            'has_hallucination': has_hallucination
        }


class HallucinationEvaluator:
    """Evaluate hallucination across multiple generations"""
    
    def __init__(self):
        """Initialize hallucination evaluator"""
        self.detector = HallucinationDetector()
        self.results = []
    
    def evaluate(self,
                generated_answer: str,
                context_chunks: List[Dict[str, Any]],
                query: str = None) -> Dict[str, Any]:
        """
        Evaluate a single generation
        
        Args:
            generated_answer: Generated answer
            context_chunks: Retrieved context
            query: Optional query text
            
        Returns:
            Hallucination metrics
        """
        result = self.detector.comprehensive_hallucination_check(
            generated_answer,
            context_chunks
        )
        
        result['query'] = query
        result['answer'] = generated_answer
        
        self.results.append(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hallucination statistics across all evaluations"""
        if len(self.results) == 0:
            return {}
        
        total = len(self.results)
        
        factual_halluc_count = sum(1 for r in self.results if r['is_factual_hallucination'])
        reasoning_halluc_count = sum(1 for r in self.results if r['is_reasoning_hallucination'])
        any_halluc_count = sum(1 for r in self.results if r['has_hallucination'])
        
        # Medical harm risk distribution
        harm_risk_dist = {
            'none': sum(1 for r in self.results if r['medical_harm_risk'] == 'none'),
            'low': sum(1 for r in self.results if r['medical_harm_risk'] == 'low'),
            'medium': sum(1 for r in self.results if r['medical_harm_risk'] == 'medium'),
            'high': sum(1 for r in self.results if r['medical_harm_risk'] == 'high')
        }
        
        # Average semantic similarity
        avg_similarity = np.mean([r['semantic_similarity'] for r in self.results])
        
        return {
            'total_evaluations': total,
            'factual_hallucination_rate': factual_halluc_count / total,
            'reasoning_hallucination_rate': reasoning_halluc_count / total,
            'overall_hallucination_rate': any_halluc_count / total,
            'avg_semantic_similarity': avg_similarity,
            'medical_harm_risk_distribution': harm_risk_dist
        }
    
    def reset(self):
        """Reset evaluation results"""
        self.results = []


if __name__ == "__main__":
    # Test hallucination detector
    detector = HallucinationDetector()
    
    # Test case 1: Grounded answer
    context = [
        {'chunk_id': 'c1', 'text': 'Patient has chest pain. Doctor recommends ECG test.'}
    ]
    answer = "The doctor recommends an ECG test for the chest pain."
    
    result = detector.comprehensive_hallucination_check(answer, context)
    
    print("Test Case 1: Grounded Answer")
    print(f"  Factual hallucination: {result['is_factual_hallucination']}")
    print(f"  Semantic similarity: {result['semantic_similarity']:.3f}")
    print(f"  Medical harm risk: {result['medical_harm_risk']}")
    
    # Test case 2: Hallucinated answer
    answer2 = "The patient needs immediate surgery and should take high doses of aspirin."
    result2 = detector.comprehensive_hallucination_check(answer2, context)
    
    print("\nTest Case 2: Potentially Hallucinated Answer")
    print(f"  Factual hallucination: {result2['is_factual_hallucination']}")
    print(f"  Semantic similarity: {result2['semantic_similarity']:.3f}")
    print(f"  Medical harm risk: {result2['medical_harm_risk']}")
    print(f"  Gives medical advice: {result2['gives_medical_advice']}")
