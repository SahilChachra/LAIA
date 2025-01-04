import os
import json
import time
import signal
import threading
from datetime import datetime
from typing import List, Dict, Any, Tuple

import redis
from loguru import logger

# Configure logging
logger.add("./logs/worker.log", rotation="10 MB", level="INFO")

class WorkerConfig:
    """Configuration for the worker process"""
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    QUEUE_NAME = 'question_queue'
    RESULT_HASH = 'result_hash'
    
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 3))
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # seconds

class RedisQueueWorker:
    """
    A robust worker class for processing tasks from a Redis queue
    
    Handles:
    - Batch processing
    - Error handling
    - Graceful shutdown
    - Logging
    """
    
    def __init__(self, llm_model, config: WorkerConfig = WorkerConfig()):
        """
        Initialize the worker
        
        Args:
            llm_model: The language model for processing
            config: Worker configuration
        """
        self.llm_model = llm_model
        self.config = config
        
        # Redis connection
        self.redis_conn = redis.Redis(
            host=self.config.REDIS_HOST, 
            port=self.config.REDIS_PORT, 
            db=self.config.REDIS_DB
        )
        
        # Shutdown flag
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def _extract_batch_data(self, data_list: List[bytes]) -> List[Tuple[str, str, int]]:
        """
        Extract and validate batch data
        
        Args:
            data_list: List of raw Redis queue data
        
        Returns:
            List of validated batch items
        """
        batch = []
        for data in data_list:
            try:
                # Parse JSON data
                parsed_data = json.loads(data)
                
                # Validate required fields
                question = parsed_data.get("question")
                request_id = parsed_data.get("request_id")
                context = parsed_data.get("context")
                history = parsed_data.get("history")
                
                if not question or not request_id:
                    logger.warning(f"Incomplete data: {parsed_data}")
                    continue
                
                # Default max sequence length
                max_seq_length = parsed_data.get("max_seq_length", 2048)
                
                batch.append((request_id, question, context, history, max_seq_length))
                
                logger.info(f"[QUEUE] Processing request ID: {request_id}")
            
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON: {data}")
            except Exception as e:
                logger.error(f"Error processing queue item: {e}")
        
        return batch

    def process_batch(self, batch: List[Tuple[str, str, str, int]]) -> List[Dict[str, str]]:
        """
        Process a batch of requests
        
        Args:
            batch: List of (request_id, prompt, max_seq_length)
        
        Returns:
            Processed batch results
        """
        try:
            # Separate batch components
            request_ids = [item[0] for item in batch]
            prompts = [item[1] for item in batch]
            contexts = [item[2] for item in batch]
            history = [item[3] for item in batch]
            max_seq_length = batch[0][2]  # Assume consistent max length

            logger.info(f"request_ids : {request_ids}, history : {history}")
            
            # Perform batch processing
            result_batch = self.llm_model.process_batch(
                request_ids=request_ids, 
                prompts=prompts, 
                contexts=contexts,
                history=history,
                max_seq_length=max_seq_length,
                batch_size=len(batch)
            )
            
            return result_batch
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Create error results for each request
            return [
                {
                    "request_id": req_id, 
                    "result": f"Processing failed: {str(e)}"
                } 
                for req_id, _, _ in batch
            ]

    def store_results(self, results: List[Dict[str, str]]):
        """
        Store batch results in Redis
        
        Args:
            results: List of processed results
        """
        for result in results:
            request_id = result.get("request_id")
            processed_result = result.get("result", "No result")
            
            try:
                self.redis_conn.hset(
                    self.config.RESULT_HASH, 
                    request_id, 
                    processed_result
                )
                logger.info(f"[RESULT] Stored result for ID: {request_id}")
            
            except Exception as e:
                logger.error(f"Failed to store result for {request_id}: {e}")

    def process_queue(self):
        """
        Continuously process items from the Redis queue
        """
        retry_count = 0
        
        while self.running:
            try:
                # Fetch batch from Redis queue
                queue_items = [
                    self.redis_conn.rpop(self.config.QUEUE_NAME) 
                    for _ in range(self.config.BATCH_SIZE)
                ]
                
                # Remove None values
                queue_items = [item for item in queue_items if item]
                
                if not queue_items:
                    # No items in queue, wait briefly
                    time.sleep(1)
                    retry_count = 0
                    continue
                
                # Extract and validate batch data
                batch = self._extract_batch_data(queue_items)
                
                if not batch:
                    continue
                
                # Process batch
                results = self.process_batch(batch)
                
                # Store results
                self.store_results(results)
                
                # Reset retry count on successful processing
                retry_count = 0
            
            except redis.exceptions.ConnectionError:
                retry_count += 1
                logger.error(f"Redis connection error. Retry {retry_count}")
                
                if retry_count > self.config.MAX_RETRY_ATTEMPTS:
                    logger.critical("Max retry attempts reached. Exiting.")
                    break
                
                time.sleep(self.config.RETRY_DELAY)
            
            except Exception as e:
                logger.error(f"Unexpected error in queue processing: {e}")
                time.sleep(1)

    def start(self):
        """
        Start the worker process
        """
        logger.info("Worker started. Waiting for tasks...")
        self.process_queue()

    def shutdown(self, signum=None, frame=None):
        """
        Graceful shutdown handler
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("Shutting down worker...")
        self.running = False

def main():
    """
    Main entry point for the worker
    """
    try:
        # Initialize LLM (replace with your actual model initialization)
        from laiaworker import LLM
        laia = LLM(
            model_name="Llama-3.2-3B-instruct", 
            model_path="../Llama-3.2-3B-Instruct-Q8_0.gguf"
        )
        
        # Create and start worker
        worker = RedisQueueWorker(laia)
        worker.start()
    
    except Exception as e:
        logger.critical(f"Worker initialization failed: {e}")

if __name__ == '__main__':
    main()