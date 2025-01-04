from datetime import datetime
import torch
from typing import List, Dict
from loguru import logger

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.custom_output_parser import CustomOutputParser
output_parser = CustomOutputParser()

logger.add("./logs/models.log", rotation="10 MB")

class LLM:
    def __init__(self, model_name, model_path="Llama-3.2-3B-Instruct-Q8_0.gguf", n_ctx=8096, 
                 n_batch=8, f16_kv=True, n_threads=8, low_vram=False, verbose=True):
        """
        Initializes the GGUF model.
        """
        self.__model_path__ = model_path
        self.__model_name__ = model_name
        self.__n_ctx__ = n_ctx
        self.__f16_kv__ = f16_kv
        self.__n_threads__ = n_threads
        self.__low_vram__ = low_vram
        self.__verbose__ = verbose
        self.__n_batch__ = n_batch
        
        try:
            self.__llm__ = LlamaCpp(
                model_path=self.__model_path__,  # Path to the GGUF model
                n_ctx=self.__n_ctx__,            # Context window size
                n_parts=-1,                       # Model splitting (set -1 to auto-configure)
                f16_kv=self.__f16_kv__,          # Use 16-bit precision for key-value cache
                n_threads=self.__n_threads__,     # Number of threads to use
                n_batch=self.__n_batch__,         # Batch size for tokens
                verbose=self.__verbose__          # Enable verbose output for debugging
            )

            logger.info(f"{self.__model_name__} loaded successfully!")  # Now this will work
        
        except Exception as e:
            logger.error(f"Error loading model {self.__model_name__}: {e}")
            self.__llm__ = None  # Set to None if loading fails

    def create_prompt_template(self, template: str) -> PromptTemplate:
        """
        Create a prompt template for consistent formatting
        
        Args:
            template (str): Prompt template string
        
        Returns:
            PromptTemplate: Configured prompt template
        """
        return PromptTemplate.from_template(template)

    def create_chain(self, prompt_template: PromptTemplate):
        """
        Create a processing chain for inference
        
        Args:
            prompt_template (PromptTemplate): Prompt template to use
        
        Returns:
            Runnable chain for processing
        """
        chain = (
            {"history" : RunnablePassthrough(), "context" : RunnablePassthrough(), "input": RunnablePassthrough()}
            | prompt_template
            | self.__llm__
            | StrOutputParser()
        )
        return chain

    def process_batch(self, 
                      request_ids: List[str], 
                      prompts: List[str], 
                      contexts: List[str],
                      history: List[str],
                      max_seq_length: int, 
                      batch_size: int = 4) -> List[Dict[str, str]]:
        """
        Process a batch of requests using LangChain
        
        Args:
            request_ids (List[str]): List of unique request identifiers
            prompts (List[str]): List of input prompts
            max_seq_length (int): Maximum sequence length
            batch_size (int, optional): Batch processing size. Defaults to 4.
        
        Returns:
            List[Dict[str, str]]: Processed results with request IDs
        """
        try:
            # Create a prompt template
            prompt_template = self.create_prompt_template(
                """
                <|begin_of_text|>
                <|start_header_id|>system<|end_header_id|>
                Provide a concise answer to the following question using best of your knowledge. You can also refer to the below history and external knowledge.

                Conversation history
                {history}

                Data from other sources as external knowledge :
                {context}
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                User Question :
                {input}
                <|start_header_id|>assistant<|end_header_id|>
                """
            )
            
            # Create the chain
            chain = self.create_chain(prompt_template)
            
            # Prepare batch inputs
            batch_inputs = [
                {"history": history, "context": context, "input": prompt} for history, context, prompt in zip(history, contexts, prompts)
            ]
            
            # Perform batch processing
            logger.info(f"Processing {len(request_ids)} requests")
            
            # Batch process the inputs
            outputs = chain.batch(
                batch_inputs, 
                batch_size=batch_size
            )
            
            # Prepare results
            results = []
            for request_id, output in zip(request_ids, outputs):
                logger.info(f"Generated result for {request_id}")
                results.append({
                    "request_id": request_id,
                    "result": output
                })
            
            return results

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            # Return error messages for all request IDs
            return [
                {
                    "request_id": request_id, 
                    "result": f"Error running inference: {str(e)}"
                } 
                for request_id in request_ids
            ]

def main():
    processor = LLM(model_name="Llama-3.2-3B-instruct", model_path="../Llama-3.2-3B-Instruct-Q8_0.gguf")
    
    # Example batch processing
    request_ids = ["req1", "req2", "req3", "req4"]
    prompts = [
        "Explain quantum computing",
        "What is the capital of France?",
        "Describe machine learning briefly",
        "What are the main causes of climate change?"
    ]
    contexts = [
        "Explain quantum computing",
        "What is the capital of France?",
        "Describe machine learning briefly",
        "What are the main causes of climate change?"
    ]
    history = [
        "Assistant : Hi! How are you User: Hi! I am good",
        "Assistant : Hi! How are you User: Hi! I am good",
        "Assistant : Hi! How are you User: Hi! I am good",
        ""
    ]
    
    results = processor.process_batch(
        request_ids=request_ids, 
        prompts=prompts,
        contexts=contexts,
        history=history,
        max_seq_length=4096,
        batch_size=4
    )
    
    # Print results
    for result in results:
        print(f"Request ID: {result['request_id']}")
        print(f"Result: {result['result']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
