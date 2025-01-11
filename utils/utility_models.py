from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import re

logger.add("./logs/utility_models.log", rotation="10 MB")

class HuggingFaceReRankerLLM:
    def __init__(self, checkpoint: None, device="cuda"):
        if not checkpoint:
            self.checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
        else:
            self.checkpoint = checkpoint
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        
    def __call__(self, user_input: str) -> str:
        """Generate text based on a user_input."""
        try:
            # Ensure user_input is a string and not None
            if user_input is None:
                logger.error("Received None user_input")
                return "NO"
            
            # Convert user_input to string explicitly
            user_input = str(user_input).strip()
            
            # Ensure user_input is not empty
            if not user_input:
                logger.error("Received empty user_input")
                return "NO"
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that re-ranks data from a database."
                },
                {
                    "role": "user", 
                    "content": f"Please re-rank this data. This is the user query {user_input}"
                }
            ]
            input_text=self.tokenizer.apply_chat_template(messages, tokenize=False)

            # Tokenize the user_input
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                add_special_tokens=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,  # Short response
                temperature=0.2, 
                top_p=0.9, 
                do_sample=True
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # For boolean parsers, explicitly return YES or NO
            response = response.upper()
            if "YES" in response:
                return "YES"
            elif "NO" in response:
                return "NO"
            
            # default response
            return "NO"
        
        except Exception as e:
            logger.error(f"Error in HuggingFaceReRankerLLM generation: {e}")
            import traceback
            traceback.print_exc()
            return "NO"


class HuggingFaceSummarizerLLM:
    def __init__(self, checkpoint=None, device="cuda"):
        if checkpoint is None:
            checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
        
        self.checkpoint = checkpoint
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        
    def clean_output(self, text: str) -> str:
        """
        Clean the output text by removing common artifacts.
        """
        # Remove common prefixes/suffixes
        artifacts = [
            "assistant", "Assistant:", "assistant:", "|')'assistant", "')'",
            "Please summarize", "following text", "Here's a summary",
            "The summary is", "Summary:"
        ]
        
        # Convert text to lowercase for checking
        text_lower = text.lower()
        
        # Remove each artifact
        for artifact in artifacts:
            if artifact.lower() in text_lower:
                text = re.sub(re.escape(artifact), '', text, flags=re.IGNORECASE)
        
        # Remove any leading/trailing punctuation and whitespace
        text = text.strip(':|.,()-\n\t ')
        
        return text.strip()
        
    def __call__(self, user_input: str) -> str:
        """
        Generate a summary of the provided text.
        
        Args:
            user_input (str): The text to be summarized
            
        Returns:
            str: The generated summary only (without any artifacts)
        """
        try:
            # Convert user_input to string and validate
            user_input = str(user_input).strip()
            if not user_input:
                logger.error("Received empty user_input")
                return ""
            
            # Create a simple summarization prompt
            messages = [
                {
                    "role": "system",
                    "content": "Provide a concise summary."
                },
                {
                    "role": "user",
                    "content": f"Summarize: {user_input}"
                }
            ]
            
            # Apply chat template and store the prompt length
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_tokens = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
            prompt_length = len(prompt_tokens.input_ids[0])
            
            # Tokenize with appropriate settings for summarization
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate summary
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                min_new_tokens=30,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Extract only the newly generated tokens
            new_tokens = outputs[0][prompt_length:]
            
            # Decode only the new tokens
            summary = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean the output
            summary = self.clean_output(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in HuggingFaceSummarizerLLM generation: {e}")
            import traceback
            traceback.print_exc()
            return ""