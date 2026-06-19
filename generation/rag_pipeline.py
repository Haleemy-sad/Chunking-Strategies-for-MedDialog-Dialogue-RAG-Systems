"""
RAG Pipeline for Medical Dialogue Generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
import time


class MedicalRAGPipeline:
    """RAG pipeline for medical question answering"""
    
    def __init__(self,
                 model_name: str = "google/flan-t5-small",
                 device: str = None,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 load_in_4bit: bool = False):
        """
        Initialize RAG pipeline
        
        Args:
            model_name: Name of the LLM model
            device: Device to use ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            load_in_4bit: Whether to use 4-bit quantization (for large models like Mistral-7B)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading LLM: {model_name} on {self.device}")
        if load_in_4bit:
            print("Using 4-bit quantization for efficient memory usage")
        
        # Load model and tokenizer
        try:
            from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure 4-bit quantization if requested
            quantization_config = None
            if load_in_4bit and self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Use appropriate model class based on model type
            if "t5" in model_name.lower() or "flan" in model_name.lower():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                # For causal LM models (Mistral, GPT, etc.)
                model_kwargs = {
                    "low_cpu_mem_usage": True,
                    # "trust_remote_code": Deprecated - removed
                }
                
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = torch.float16 if self.device == 'cuda' else torch.float32
                    if self.device == 'cuda':
                        model_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            
            if self.device == 'cpu' and not load_in_4bit:
                self.model = self.model.to(self.device)
            
            # Create pipeline with appropriate task type
            task_type = "text2text-generation" if "t5" in model_name.lower() or "flan" in model_name.lower() else "text-generation"
            
            # When using quantization with device_map="auto", don't specify device in pipeline
            pipeline_kwargs = {
                "task": task_type,
                "model": self.model,
                "tokenizer": self.tokenizer
            }
            
            if not (load_in_4bit or quantization_config):
                # Only set device if not using quantization
                pipeline_kwargs["device"] = 0 if self.device == 'cuda' else -1
            
            self.pipe = pipeline(**pipeline_kwargs)
            self.task_type = task_type
            
            print(f"LLM loaded successfully")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Falling back to mock generation for testing...")
            self.model = None
            self.tokenizer = None
            self.pipe = None
        
        self.stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'avg_generation_time': 0,
            'avg_tokens_per_second': 0
        }
    
    def create_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Create RAG prompt"""

        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
            
        prompt = f"""You are a medical AI assistant. Based on the medical knowledge provided below, answer the patient's question.

        Medical Knowledge:
        {context}

        Patient Question: {query}

        Instructions:
        - Provide a general medical answer based on the knowledge above
        - Do NOT refer to specific patient cases (avoid "in your case")
        - Give factual medical information about possible causes, symptoms, and general advice
        - Keep the answer concise (2-3 sentences)
        - If uncertain, recommend consulting a healthcare provider

        Answer:"""
    
        return prompt
    
    def generate(self, 
                query: str,
                retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using RAG
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Create prompt
        prompt = self.create_prompt(query, retrieved_chunks)
        
        # Generate
        if self.pipe is not None:
            try:
                # Generate using pipeline
                output = self.pipe(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
                
                # Extract generated text
                generated_text = output[0]['generated_text']
                
                # Extract only the answer (after "ANSWER:")
                if "ANSWER:" in generated_text:
                    answer = generated_text.split("ANSWER:")[-1].strip()
                else:
                    answer = generated_text.strip()
                
                # Count tokens
                tokens_generated = len(self.tokenizer.encode(answer))
                
            except Exception as e:
                print(f"Error during generation: {e}")
                answer = self._mock_generate(query, retrieved_chunks)
                tokens_generated = len(answer.split())
        
        else:
            # Mock generation for testing
            answer = self._mock_generate(query, retrieved_chunks)
            tokens_generated = len(answer.split())
        
        # Calculate generation time
        generation_time = time.time() - start_time
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Update stats
        self.stats['total_generations'] += 1
        self.stats['total_tokens_generated'] += tokens_generated
        self.stats['avg_generation_time'] = (
            (self.stats['avg_generation_time'] * (self.stats['total_generations'] - 1) + generation_time)
            / self.stats['total_generations']
        )
        self.stats['avg_tokens_per_second'] = (
            (self.stats['avg_tokens_per_second'] * (self.stats['total_generations'] - 1) + tokens_per_second)
            / self.stats['total_generations']
        )
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'num_chunks': len(retrieved_chunks),
            'tokens_generated': tokens_generated,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'prompt': prompt
        }
    
    def generate_from_context(self, query: str, context: str, max_tokens: int = 300) -> str:
        """
        Generate answer from query and context string
        
        Args:
            query: User query
            context: Context string
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer text
        """
        # Truncate context if too long (T5 has 512 token limit)
        if self.tokenizer is not None:
            context_tokens = self.tokenizer.encode(context)
            if len(context_tokens) > 350:  # Leave room for question and answer
                context = self.tokenizer.decode(context_tokens[:350], skip_special_tokens=True)
        else:
            # Simple truncation by characters if tokenizer not available
            if len(context) > 1500:
                context = context[:1500]
        
        # Medical-focused prompt that works well with flan-t5-base
        prompt = f"""Based on the medical information below, provide a clear and informative answer to the question.

Medical Information:
{context}

Question: {query}

Provide a helpful medical answer (2-3 sentences):"""
        
        if self.pipe is not None:
            try:
                print(f"Generating answer for: {query[:50]}...")
                
                # Generate using pipeline
                if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                    # T5 models use text2text generation
                    # max_length is the total output length, not additional tokens
                    output = self.pipe(
                        prompt, 
                        max_length=250,  # Total output length
                        min_length=30,   # Minimum response length
                        num_return_sequences=1,
                        clean_up_tokenization_spaces=True,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        repetition_penalty=2.0,  # Penalize repetition
                        no_repeat_ngram_size=3   # Prevent 3-word repetitions
                    )
                    answer = output[0]['generated_text'].strip()
                    
                    print(f"Raw model output: {answer[:200]}")
                    
                    # T5 sometimes returns the entire prompt - extract only the answer part
                    # Look for "Answer:" marker and extract everything after it
                    if "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    
                    # If the answer contains the question, extract what comes after
                    if query in answer:
                        parts = answer.split(query)
                        if len(parts) > 1:
                            answer = parts[-1].strip()
                            # Remove any leading colons or spaces
                            answer = answer.lstrip(': ')
                    
                    # If answer contains "Context:" it returned the whole prompt
                    if "Context:" in answer:
                        # Try to extract just the answer part
                        if "Answer:" in answer:
                            answer = answer.split("Answer:")[-1].strip()
                        else:
                            # Model failed, use a simple extraction from context
                            print("Model returned full prompt, extracting from context...")
                            # Just take the first sentence from the first retrieved chunk
                            sentences = context.split('.')[:3]
                            answer = '. '.join(sentences) + '.'
                    
                    # Clean up any remaining artifacts
                    if len(answer) < 10:
                        print("Answer too short, using context summary")
                        answer = "Based on the medical information: " + context[:200] + "..."
                    
                    # Post-process to remove hallucination/repetition
                    answer = self._clean_answer(answer, query)
                    
                    print(f"Generated answer: {answer[:100]}...")
                    print(f"Answer length: {len(answer)}")
                else:
                    # Causal LM models
                    output = self.pipe(
                        prompt,
                        max_new_tokens=min(max_tokens, self.max_new_tokens),
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True
                    )
                    generated_text = output[0]['generated_text']
                    # Extract answer after the prompt
                    answer = generated_text[len(prompt):].strip()
                    print(f"Generated answer length: {len(answer)}")
                
                if not answer or len(answer.strip()) == 0:
                    return "Unable to generate a specific answer from the provided context. Please review the retrieved chunks above for relevant information."
                
                return answer
                
            except Exception as e:
                error_msg = f"Error during generation: {str(e)}"
                print(error_msg)
                return error_msg
        else:
            return "Model not loaded. Retrieved context: " + context[:300] + "..."
    
    def _clean_answer(self, answer: str, query: str) -> str:
        """Clean and validate the generated answer"""
        
        # Remove if answer is just repeating the query
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Check if answer is mostly repeating the question
        query_words = set(query_lower.split())
        answer_words = answer_lower.split()
        
        # If >70% of answer words are from the query, it's likely repetition
        if len(answer_words) > 5:
            overlap = sum(1 for word in answer_words if word in query_words)
            overlap_ratio = overlap / len(answer_words)
            
            if overlap_ratio > 0.7:
                return "I apologize, but I cannot provide a specific answer based on the available medical context. The symptoms you describe (restless legs, lump, fatigue during pregnancy) should be evaluated by a healthcare professional. Please consult your doctor for proper diagnosis and treatment."
        
        # Remove repetitive phrases
        sentences = answer.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen:
                seen.add(sentence.lower())
                unique_sentences.append(sentence)
        
        cleaned_answer = '. '.join(unique_sentences)
        if cleaned_answer and not cleaned_answer.endswith('.'):
            cleaned_answer += '.'
        
        return cleaned_answer if cleaned_answer else answer
    
    def _mock_generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Mock generation for testing without LLM
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
            
        Returns:
            Mock answer
        """
        # Extract key information from chunks
        chunk_texts = [chunk['text'] for chunk in retrieved_chunks]
        combined = " ".join(chunk_texts)
        
        # Simple mock answer
        answer = f"Based on the provided context: {combined[:200]}..."
        
        return answer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'avg_generation_time': 0,
            'avg_tokens_per_second': 0
        }


class RAGSystem:
    """Complete RAG system combining retrieval and generation"""
    
    def __init__(self, retriever, generator: MedicalRAGPipeline):
        """
        Initialize RAG system
        
        Args:
            retriever: Retriever instance
            generator: MedicalRAGPipeline instance
        """
        self.retriever = retriever
        self.generator = generator
    
    def query(self, 
             question: str,
             top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks, scores = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate answer
        result = self.generator.generate(question, retrieved_chunks)
        
        # Add retrieval scores
        result['retrieval_scores'] = scores
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        return {
            'retrieval': self.retriever.get_stats(),
            'generation': self.generator.get_stats()
        }


if __name__ == "__main__":
    # Test RAG pipeline
    print("Testing RAG Pipeline with mock generation...")
    
    # Create dummy chunks
    chunks = [
        {'chunk_id': 'chunk_0', 'text': 'Patient: I have chest pain. Doctor: Can you describe the pain?'},
        {'chunk_id': 'chunk_1', 'text': 'Patient: It is a dull, squeezing pain. Doctor: I recommend an ECG test.'},
    ]
    
    # Create pipeline
    pipeline = MedicalRAGPipeline()  # Will use mock generation
    
    # Test generation
    query = "What test does the doctor recommend?"
    result = pipeline.generate(query, chunks)
    
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Generation time: {result['generation_time']:.3f}s")
    print(f"Tokens generated: {result['tokens_generated']}")
    
    print(f"\nStats: {pipeline.get_stats()}")
