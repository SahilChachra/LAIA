# LAIA - Language AI Advisor 📚🤖

Welcome to **LAIA** (Language AI Advisor), your modular AI companion!

---

## 📋 Description

LAIA (Language AI Advisor) is a modular server built using Large Language Models (LLMs) with tools for PDF-based Retrieval-Augmented Generation (RAG), web search, and conversation history management. The project aims to create a flexible and scalable framework for integrating various AI tools as needed.

## 💡 Design Philosophy

The goal behind this project was to build an **LLM server with custom agents** that is:

1. **Open and Flexible**: Designed to avoid dependencies on specific models or libraries, making it highly adaptable to various use cases.  
2. **Modular Architecture**: The code is structured to be as modular as possible, allowing it to serve as a foundational base for similar projects or future extensions.  
3. **Built from Scratch**: While many open-source libraries exist, this project was built independently to ensure that individual components can function autonomously, promoting scalability and reusability.

This approach ensures that LAIA is not only powerful but also extensible, encouraging innovation and customization.

---

## 🌠 Screenshots

![Screenshot 1](https://github.com/SahilChachra/LAIA/blob/main/images/LAIA_SS_1.png)

![Screenshot 2](https://github.com/SahilChachra/LAIA/blob/main/images/LAIA_SS_2.png)


---

## 🧠 Models Details
### Memory usage details :
- **Base Load**: 3.5 GB
  - Includes 2 instances of SmolLMv2 and 1 instance of Llama-3.2-3B-instruct_Q8.gguf
- **Ingestion Process**: 5 GB
  - Required during `/ingest` operations for vector database and re-ranking models
- **Peak Usage**: 5.8 GB
  - Observed during response generation
- **Generation Speed**: ~18 tokens/second
### Models
1. **[Llama-3.2-3B-instruct_Q8.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)**: Primary LLM used to generate responses.  
2. **thenlper/gte-small**: Embedding model for creating vectorized representations in the vector store (**ChromaDB**).  
3. **SmolLMv2-360M-instruct**: Lightweight LLM used to filter and select relevant outputs from **ChromaDB** and also to summarize data.  
4. **BAAI/bge-small-en**: Cross-encoder used for re-ranking documents for precision in final outputs.  
### Model path
I had place the Llama-3.2-3B-instruct model in the parent folder of LAIA. You can do the same or update the path in ```worker.py```'s main() accordingly. 

---

## 🚀 Key Features

1. **Advanced RAG with Multi-Level Filtering**:  
   LAIA uses a robust three-step document retrieval process:  
   - **Cosine Similarity**: Quickly fetches the most relevant documents from the **ChromaDB** vector database.  
   - **LLMChainFilter**: Filters out irrelevant documents using the **SmolLMv2-360M-instruct model**, ensuring only relevant content reaches the next stage.  
   - **Cross-Encoder Reranker**: Re-ranks documents for precision using **BAAI/bge-small-en**, fine-tuning results for accuracy.  

2. **Web Search Integration**:  
   Combines knowledge from web searches allowing it to write well-rounded answers for user queries.

3. **Powered by Cutting-Edge LLMs**:  
   - Utilizes the **Llama-3.2-3B-Instruct-Q8_0.gguf model** to generate detailed and insightful answers.  
   - Efficient inference with Redis-powered batch processing ensures smooth multi-user support.

4. **Optimized for Performance**:  
   - Built with **Langchain** for seamless integration of components.  
   - Uses **Redis** for asynchronous request handling, enabling batch inference and improving responsiveness.  
   - Deployed using **FastAPI** and **Gunicorn**, ensuring scalable and robust performance.

---

## Flow Charts (Build using lucid.app)

### Web Seach Flow
![Web Search flow](https://github.com/SahilChachra/LAIA/blob/main/images/WebSearchToolFlowChart.png)

### RAG flow

## 🛠️ Technologies Used

- **[Langchain](https://github.com/hwchase17/langchain)**: Framework for building LLM applications.  
- **[ChromaDB](https://docs.trychroma.com/)**: Manages and retrieves vectorized document embeddings for RAG.  
- **[Redis](https://redis.io/)**: Enables efficient multi-user support and batch processing.  
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance API framework for serving LAIA.  
- **[Gunicorn](https://gunicorn.org/)**: Python WSGI server for deploying the application.

---

## 💻 Setup and Usage

### 1. Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/LAIA.git
cd LAIA
pip install -r requirements.txt
```

### 1.X Better approach
Pull an image from Nvidia with CUDA preinstalled with Ubuntu 22.04. Install PyTorch with CUDA, Llamacpp with CUDA support and relevant packages from requirements. It's a bit tricky to setup but carries a huge learning curve.

Copy or clone the code inside the container and the steps to run the code remains the same.

### 2. Starting the Server

Launch the server, Redis, and the worker in separate terminals:

1. **Start the FastAPI server**:
   ```bash
   gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:80 --timeout 10000 --access-logfile guni.log
   ````
2. **Start Redis**:
    ```bash
    redis-server
    ````
3. **Start the Worker**
    ```bash
    python3 worker.py
    ```
5. **Modify client_ingest.py and client_infer.py to use the server**
    ```bash
    python3 client_ingest.py # If you want to use RAG

    python3 client_infer.py # Enable/Disable RAG and Websearch tool
    ```

## 2. Usage

Make sure you get your own Serper api key from [Serper's site](https://serper.dev/) to use Web Search tool.
Create a .env file in the project folder and paste it like
```bash
SERPER_API_KEY=your_api_key_here
```

client_ingest.py - If you want to use RAG with your PDF, you can run this code first which will ingest your data to a Vector database and return a PDF RAG Tool against your user id.

client_infer.py - You can modify the payload value and turn on/off the RAG and web search tool.

NOTE - Your PDF RAG Tool is saved against your user id. Make sure you have same user id in Ingest and Infer.

---

## 🛠️ Future Roadmap
 
- **Improved Language Model Support**: Add more advanced LLMs for broader language capabilities.    
- **Expanded File Support**: Add support for different file types, including PowerPoint presentations and spreadsheets.  
- **UI Development**: Build a user-friendly web interface to enhance accessibility.
- **Code Parsers**: Add code parsers to handle Python/C++ code as output

---

## 📜 License

This project is licensed under the [GPL-3.0 License](LICENSE).

---

## 🙌 Acknowledgments

Special thanks to the creators of:
- **[Langchain](https://github.com/hwchase17/langchain)** for simplifying LLM integrations.  
- **[ChromaDB](https://docs.trychroma.com/)** for efficient vector storage and retrieval.  
- **[Hugging Face](https://huggingface.co/)** for their open-source models like **BAAI/bge-small-en** and **SmolLMv2-360M-instruct**.

---

## 📩 Contact

For issues, questions, or contributions, feel free to open an issue or contact [sahil.chachra3@live.com](mailto:sahil.chachra3@live.com).
