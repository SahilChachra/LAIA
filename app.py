import json
import time
from datetime import datetime
from typing import Dict, Any
import asyncio
import redis
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from typing import Optional, List
from sentence_transformers import CrossEncoder
from pathlib import Path
import dill

from utils.pdf_agent import PDFAgent
from utils.web_search_agent import WebSearchTool
from utils.process_text_data import process_data
from utils.store_history import ConversationDatabase
from utils.utility_models import HuggingFaceSummarizerLLM

from dotenv import load_dotenv
import os

load_dotenv()
try:
    serper_api_key = os.getenv("SERPER_API_KEY")
    logger.success("SERPER_API_KEY found")
except Exception as e:
    logger.error("Couldn't load SERPER_API_KEY from .env file! Please re-verify!")

# Configure logging
logger.add("./logs/server.log", rotation="10 MB")

# Redis connection
redis_conn = redis.Redis(host="localhost", port=6379, db=0)

# Initialize the database to store history
try:
    db_history = ConversationDatabase()
    db_history.clear_table()
    logger.success("Database connected successfully and cleared!")
except:
    logger.warning("Database connection failed!")

try:
    history_summarizer = HuggingFaceSummarizerLLM(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", device="cuda")
    logger.success("Conversation History summarizer model loaded successfully!")
except:
    logger.warning("History summarizer model failed to load!")


# Declare tools as global
try:
    web_search_tool = WebSearchTool(api_key=serper_api_key)
    logger.success("Web Search tool created successfully!")
except Exception as e:
    logger.error("Error creating Web Search tool : {e}")
pdf_retrieval_tools: Dict[str, object] = {}

# ReRanker model
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# FastAPI app
app = FastAPI(
    title="Inference Service",
    description="Real-time inference service with Redis queue management",
    version="1.0.0"
)

logger.success("All settings completed. Starting App...")

# Request model for input validation
class InferenceRequest(BaseModel):
    question: str
    request_id: str
    use_pdf_rag: bool
    use_web_search: bool
    user_id: str

# Define IngestionRequest model
class IngestionRequest(BaseModel):
    files: list[UploadFile]
    chunk_overlap: int
    chunk_size: int
    request_id: str  # Changed to str for UUID-like IDs
    user_id: str
    embedding_model_name: Optional[str] = "thenlper/gte-small"  # Optional with default

# Dependency function to retrieve the pdf_retrieval_tool by user_id
def get_pdf_retrieval_tool(request: InferenceRequest):
    if not request.use_pdf_rag:
        logger.info("User is not using PDF Retrieval for this request")
        return None
    
    print(f"{request.user_id} : {pdf_retrieval_tools}")
    pdf_retrieval_tool = pdf_retrieval_tools.get(request.user_id)
    if not pdf_retrieval_tool:
        raise HTTPException(status_code=404, detail="PDF Retrieval Tool not found")
    logger.success(f"PDF Retrieval tool found for user : {request.user_id}")
    return pdf_retrieval_tool

@app.post("/generate")
async def infer_text(request: InferenceRequest, pdf_retrieval_tool: object = Depends(get_pdf_retrieval_tool)):
    """
    Endpoint for submitting inference requests
    
    Args:
        request (InferenceRequest): Inference request with question and request_id
    
    Returns:
        JSONResponse with inference result
    """
    try:
        # Log incoming request
        logger.info(f"[INFO] {datetime.now()} | Request for Id: {request.request_id} in Queue")
        
        # Clean the input question
        question = process_data(request.question)

        retrieval_results = ""
        web_search_results = ""

        if (not request.use_pdf_rag) and (not request.use_web_search):
            logger.warning(f"Request {request.request_id} - Both RAG and Web seach tools are disabled! Model will generate answer from its own knowledge!")

        if request.use_pdf_rag:
            try:
                retrieval_results = pdf_retrieval_tool.func(question)
                logger.info(f"PDF Retrieval results : {retrieval_results}")
            except:
                logger.error(f"Error from PDF Retrieval tool : {request.user_id}")
        
        # Use Web search
        if request.use_web_search:
            web_search_results = web_search_tool.run(question)
            # logger.info(f"Web search results : {web_search_results}")
        
        # Fetch conversation history
        history = db_history.fetch_history(request.user_id)
        if not history:
            history = ""
        else:
            try:
                history = history_summarizer(history)
                logger.success(f"Conversation history summarized!")
            except Exception as e:
                logger.warning(f"Conversation history was not summarized!")
            logger.info(f"Conversation history : User_id : {request.user_id} - {history}")

        combined_context = f"""
        Retrieved Documents:
        {retrieval_results}

        Web Search Results:
        {web_search_results}
        """

        # Prepare data for Redis queue
        data = {
            "request_id": request.request_id,
            "question": request.question,
            "context": combined_context,
            "history" : history,
            "status": "processing"
        }

        # Push request to Redis queue
        redis_conn.lpush("question_queue", json.dumps(data))

        logger.info(f"{data['request_id']} pushed to queue for inference")
        
        # Poll for result with timeout
        max_wait_time = 60  # 1 minute max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check for response in result hash
            response_data = redis_conn.hget("result_hash", request.request_id)
            
            if response_data:
                # Decode and remove result from hash
                response_data = response_data.decode("utf-8")
                redis_conn.hdel("result_hash", request.request_id)
                
                # Log successful response
                logger.info(f"[INFO] {datetime.now()} | Response for Id: {request.request_id} sent")
                
                try:
                    conv_history = f"User : {request.question}, Assistant : {response_data}"
                    db_history.save_or_update_message(user_id=request.user_id, message=conv_history)
                    logger.info(f"Conversation history saved to database for user id : {request.user_id}")
                except Exception as e:
                    logger.error(f"Error saving coversation history to database! User id : {request.user_id}. Error : {e}")

                return JSONResponse(
                    status_code=200,
                    content={
                        "question": request.question,
                        "response": response_data,
                        "request_id": request.request_id,
                        "StatusCode": 200,
                        "Message": "Answer generated Successfully"
                    }
                )
            
            # Short sleep to prevent tight polling
            await asyncio.sleep(0.5)
        
        # Timeout handling
        logger.warning(f"Request {request.request_id} timed out")
        return JSONResponse(
            status_code=408,
            content={
                "request_id": request.request_id,
                "StatusCode": 408,
                "Message": "Request timed out"
            }
        )
    
    except Exception as e:
        # Comprehensive error handling
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request.request_id,
                "StatusCode": 500,
                "Message": "Internal server error",
                "Error": str(e)
            }
        )

# Additional health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status
    """
    try:
        # Check Redis connection
        redis_conn.ping()
        return {
            "status": "healthy",
            "service": "Inference Service",
            "redis": "Connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/ingest")
async def ingest_data(
    files: List[UploadFile] = File(...),
    chunk_overlap: int = Form(...),
    chunk_size: int = Form(...),
    request_id: str = Form(...),
    user_id: str = Form(...),
    embedding_model_name: Optional[str] = Form("thenlper/gte-small")
):
    """
    Ingest PDF data into the system, create embeddings, and setup retrieval tool.
    """
    # Log incoming request
    logger.info(f"[INFO] {datetime.now()} | Request for ID: {request_id} in progress.")
 
    # Define a directory to save uploaded PDFs
    base_directory = Path(__file__).parent / "uploaded_pdfs"
    base_directory.mkdir(parents=True, exist_ok=True)

    saved_file_paths = []
    errors = []

    for file in files:
        logger.info(f"[INFO] {datetime.now()} | Processing file: {file.filename}")

        # Check file type
        if not file.content_type == "application/pdf":
            logger.error(f"[ERROR] {datetime.now()} | Unsupported file type: {file.content_type}")
            errors.append({"file_name": file.filename, "error": "Unsupported file type"})
            continue

        try:
            # Save the file to disk
            file_path = base_directory / file.filename
            content = await file.read()
            
            with file_path.open("wb") as f:
                f.write(content)

            logger.info(f"[INFO] {datetime.now()} | File saved successfully: {file_path}")
            saved_file_paths.append(str(file_path))

            # Make sure to seek to start of file for any subsequent operations
            await file.seek(0)

        except Exception as e:
            logger.error(f"[ERROR] {datetime.now()} | Error saving file {file.filename}: {str(e)}")
            errors.append({"file_name": file.filename, "error": str(e)})
    
    logger.info(f"Saved file paths : {saved_file_paths}")

    # Call PDFAgent with all valid file path
    if saved_file_paths:
        try:
            logger.info(f"[INFO] {datetime.now()} | Initializing PDFAgent for {len(saved_file_paths)} files")
            pdf_agent = PDFAgent(
                embedding_model_name=embedding_model_name,
                doc_paths=saved_file_paths,  # Pass the list of file paths
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            pdf_retrieval_tools[user_id] = pdf_agent.create_retrieval_tool()

            logger.info(f"[INFO] {datetime.now()} | PDFAgent created successfully")

        except Exception as e:
            logger.error(f"[ERROR] {datetime.now()} | Error initializing PDFAgent: {str(e)}")
            return {
                "message": "Failed to process uploaded files",
                "errors": [{"error": str(e)}]
            }

    return {
        "message": "PDFs ingestion completed",
        "processed_files": saved_file_paths,
        "errors": errors
    }
        

# Optional: Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests
    """
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Outgoing response: {response.status_code}")
    return response

def main():
    """
    Main entry point to run the FastAPI server
    """
    uvicorn.run(
        "inference_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )

if __name__ == "__main__":
    main()