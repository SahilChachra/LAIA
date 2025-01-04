from utils.custom_imports import *

class CustomOutputParser:
    """
    Custom output parser to handle flexible response parsing
    """
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the raw text response into a structured format
        
        Args:
            text (str): Raw text response from the model
        
        Returns:
            Dict with answer and sources
        """
        # Default response structure
        response = {
            "answer": text,
            "sources": []
        }
        
        # Extract sources
        sources_match = re.findall(r'Sources:([^*]+)', text, re.DOTALL)
        if sources_match:
            response['sources'] = [
                source.strip() 
                for source in sources_match[0].split('\n') 
                if source.strip()
            ]
        
        return response