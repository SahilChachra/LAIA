import sys
from pathlib import Path

# Ensure you have the necessary imports from the original script
from utils.custom_imports import *

from utils.utility_models import HuggingFaceReRankerLLM

from loguru import logger
from loguru import logger
from transformers import pipeline
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from pathlib import Path

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.tools import Tool

logger.add("./logs/pdf_agent.log", rotation="10 MB")

class PDFAgent:
    def __init__(self, embedding_model_name: str, doc_paths: list[str], chunk_size: int, chunk_overlap: int, debug: bool = False):
        """
        Initialize local document retrieval and reranking with LM.
        
        Args:
            embedding_model_name (str): Path to embedding model (e.g., SentenceTransformers or LlamaCpp).
            doc_paths (list[str]): List of paths to the documents (PDF or DOCX) for retrieval.
            chunk_size (int): Maximum size of document chunks.
            chunk_overlap (int): Overlap between chunks for better context.
            debug (bool): Enable debug logging.
        """
        self.debug = debug

        # Convert paths to Path objects and resolve to absolute paths
        resolved_paths = [Path(path).resolve() for path in doc_paths]

        if self.debug:
            logger.debug(f"Resolved document paths: {resolved_paths}")
            logger.debug(f"Current working directory: {Path.cwd()}")

        # Validate all paths and collect valid document paths
        document_paths = []
        for path in resolved_paths:
            if not path.exists():
                logger.error(f"File not found: {path}")
                continue
            if path.is_file() and path.suffix.lower() in [".pdf", ".docx"]:
                document_paths.append(path)
            elif path.is_dir():
                document_paths.extend(path.glob("*.pdf"))
                document_paths.extend(path.glob("*.docx"))

        if not document_paths:
            logger.error("No valid documents found")
            raise ValueError("No valid documents found for processing")

        # Process documents
        self.documents = self._load_documents(document_paths)

        # Raise error if no content was successfully extracted
        if not self.documents:
            raise ValueError("No content could be extracted from the provided documents")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.splits = text_splitter.split_documents(self.documents)

        # Create embeddings and vector store
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.ChromaDB = Chroma.from_documents(self.splits, self.embedding_model)

        similarity_retriever = self.ChromaDB.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        logger.info("Created: similarity_retriever")

        # Apply a filter using an LLM chain
        self.huggingface_llm = HuggingFaceReRankerLLM(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", device="cuda")
        _filter = LLMChainFilter.from_llm(llm=self.huggingface_llm)
        compressor_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=similarity_retriever)
        logger.info("Created: compressor_retriever")

        self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-small-en")
        self.reranker_compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
        logger.info("Created: reranker_compressor")

        # Retriever 3 - Uses a Reranker model to rerank retrieval results
        self.final_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker_compressor,
            base_retriever=compressor_retriever
        )
        logger.info("Created: final_retriever")

    def _load_documents(self, document_paths: list[Path]) -> list:
        """
        Load and process documents from the given paths.
        
        Args:
            document_paths (list[Path]): List of document paths to process.
        
        Returns:
            list: List of loaded and processed documents.
        """
        all_documents = []
        for path in document_paths:
            logger.info(f"Processing document: {path}")
            try:
                loader = PDFPlumberLoader(str(path)) if path.suffix == ".pdf" else Docx2txtLoader(str(path))
                docs = loader.load()
                
                # Filter out empty documents
                docs = [doc for doc in docs if doc.page_content.strip()]
                
                if not docs:
                    logger.warning(f"No content extracted from {path}")
                    continue
                
                all_documents.extend(docs)
                logger.info(f"Extracted {len(docs)} documents from {path}")

            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        return all_documents

    def create_retrieval_tool(self):
        """
        Create a retrieval tool for the agent with reranking
        
        Returns:
            Tool: A custom retrieval tool
        """
        def retrieve_docs(query: str) -> str:
            """Retrieve, rerank, and format relevant documents"""
            logger.info(f"Retrieve docs() has been called by the tool!")
            docs = self.final_retriever.invoke(query)
            logger.info(f"Top {len(docs)} retrieved documents.")
            
            if(self.debug):
                logger.debug(f"Raw retreived document data : \n{docs}")

            return "\n\n".join([
                f"Document {i + 1} (Source: {doc.metadata.get('source', 'N/A')}):\n{doc.page_content[:500]}"
                for i, doc in enumerate(docs)
            ])
        
        return Tool(
            name="document_retrieval_with_reranking",
            func=retrieve_docs,
            description="Retrieve and rerank documents from the local knowledge base using a language model."
        )

def main():
    try:
        # Configure logging
        logger.remove()  # Remove default logger
        logger.add(sys.stderr, level="INFO")
        logger.add("./logs/pdf_agent_test.log", rotation="10 MB")

        # Configuration parameters
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        doc_paths = ['/home/LLMs/LAIA_v0_1/uploaded_pdfs/Relativity.pdf']
        chunk_size = 500
        chunk_overlap = 50

        # Create PDFAgent instance
        logger.info("Initializing PDFAgent!")
        pdf_agent = PDFAgent(
            embedding_model_name=embedding_model_name, 
            doc_paths=doc_paths, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

        # Create retrieval tool
        retrieval_tool = pdf_agent.create_retrieval_tool()

        test_queries = [
            "Tell me about theory of relativity.",
        ]

        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' ---")
            try:
                results = retrieval_tool.func(query)
                print(f"Query: {query}")
                print(results)
            except Exception as query_error:
                logger.error(f"Error retrieving documents for query '{query}': {query_error}")

    except Exception as e:
        logger.error(f"Error in main(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()