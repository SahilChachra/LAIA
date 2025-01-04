import requests
from bs4 import BeautifulSoup
from typing import Optional, List
import os
import json
from loguru import logger
from utils.utility_models import HuggingFaceSummarizerLLM

class WebSearchTool:
    """
    Web search tool using Serper API optimized for LLM consumption.
    This tool integrates HuggingFaceSummarizerLLM for summarizing the content.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool
        
        Args:
            api_key (str, optional): Serper API key
            summarizer (HuggingFaceSummarizerLLM, optional): Instance of HuggingFaceSummarizerLLM for summarization
        """
        self.api_key = api_key or os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("Serper API key is required. Set SERPER_API_KEY env variable.")
        
        self.base_url = "https://google.serper.dev/search"
        
        # Use the provided HuggingFaceSummarizerLLM instance for summarization
        try:
            self.summarizer = HuggingFaceSummarizerLLM(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", device="cuda")
            logger.success("HuggingFaceSummarizerLLM model loaded!")
        except:
            self.summarizer = None
            logger.warning("HuggingFaceSummarizerLLM couldn't be loaded!")
    
    def run(self, query: str, num_results: int = 3) -> str:
        """
        Perform web search, scrape URLs, summarize, and return all content as a single string.
        
        Args:
            query (str): Search query
            num_results (int, optional): Number of results to return
        
        Returns:
            str: Compiled and summarized content from search results and scraped pages
        """
        # Perform web search
        urls = self.search_web(query, num_results)
        logger.info(f"URLs : {urls}")
        # Scrape content from the URLs
        content = self.scrape_multiple_urls(urls)
        
        # Compile all the content into a single string
        compiled_content = self.compile_content(query, content)
        
        # Summarize the compiled content using HuggingFaceSummarizerLLM
        if self.summarizer:
            summarized_content = self.summarizer(compiled_content)
        else:
            logger.warning("Did not run summarizer on gathered data. Returning raw data!")
            return compiled_content
        
        return summarized_content

    def search_web(self, query: str, num_results: int) -> List[str]:
        """
        Perform web search and return a list of URLs.
        
        Args:
            query (str): Search query
            num_results (int, optional): Number of results to return
        
        Returns:
            List[str]: List of URLs
        """
        payload = json.dumps({
            "q": query,
            "num": num_results,
            "gl": "us",
            "hl": "en",
            "type": "search",
            "page": 1,
            "autocorrect": True
        })
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, data=payload)
            response.raise_for_status()
            search_results = response.json()

            logger.info(f"Query: {query}")
            
            # Collect URLs from the search results
            urls = []
            if 'organic' in search_results:
                for result in search_results['organic']:
                    urls.append(result.get('link'))
            
            return urls

        except requests.RequestException as e:
            error_msg = f"Web search error: {str(e)}"
            logger.error(error_msg)
            return []

    def scrape_url(self, url: str) -> str:
        """
        Scrape the content of a given URL.
        
        Args:
            url (str): The URL to scrape
        
        Returns:
            str: Extracted content from the page
        """
        try:
            # Send a GET request to fetch the page content
            response = requests.get(url)
            response.raise_for_status()
            
            # Use BeautifulSoup to parse the page content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the main content from the page (example: paragraphs, headers)
            paragraphs = soup.find_all('p')
            text_content = "\n".join([para.get_text() for para in paragraphs])

            # Optionally, extract other elements like title, headings, etc.
            title = soup.title.string if soup.title else "No title"
            
            # Combine title and text content
            page_content = f"Title: {title}\n\nContent:\n{text_content}"
            return page_content

        except requests.RequestException as e:
            error_msg = f"Error scraping URL {url}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def scrape_multiple_urls(self, urls: List[str]) -> List[str]:
        """
        Scrape multiple URLs and return the extracted content from each.
        
        Args:
            urls (List[str]): List of URLs to scrape
        
        Returns:
            List[str]: List of extracted content from each URL
        """
        results = []
        for url in urls:
            result = self.scrape_url(url)
            results.append(result)
        return results

    def compile_content(self, query: str, scraped_content: List[str]) -> str:
        """
        Compile the search query and scraped content into a single string for LLM consumption.
        
        Args:
            query (str): The original search query
            scraped_content (List[str]): The content scraped from the URLs
        
        Returns:
            str: The compiled content
        """
        compiled_content = f"Search Query: {query}\n\n"
        
        # Add scraped content
        for index, content in enumerate(scraped_content, start=1):
            compiled_content += f"Content from URL {index}:\n{content}\n\n"
        
        return compiled_content
