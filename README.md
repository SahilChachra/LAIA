# LAIA - Language AI Advisor 📚🤖

Welcome to **LAIA** (Language AI Advisor), your academic AI companion designed to revolutionize the way you study and learn!

---

## 💡 Design Philosophy

The goal behind this project was to build an **LLM server with agents** that is:

1. **Open and Flexible**: Designed to avoid dependencies on specific models or libraries, making it highly adaptable to various use cases.  
2. **Modular Architecture**: The code is structured to be as modular as possible, allowing it to serve as a foundational base for similar projects or future extensions.  
3. **Built from Scratch**: While many open-source libraries exist, this project was built independently to ensure that individual components can function autonomously, promoting scalability and reusability.

This approach ensures that LAIA is not only powerful but also extensible, encouraging innovation and customization.

---

## 🧠 Models Used

1. **Llama-3.2-3B-instruct_Q8.gguf**: Primary LLM used to generate responses.  
2. **thenlper/gte-small**: Embedding model for creating vectorized representations in the vector store (**ChromaDB**).  
3. **SmolLMv2-360M-instruct**: Lightweight LLM used to filter and select relevant outputs from **ChromaDB**.  
4. **BAAI/bge-small-en**: Cross-encoder used for re-ranking documents for precision in final outputs.  

---

## 🚀 Key Features

1. **Advanced RAG with Multi-Level Filtering**:  
   LAIA uses a robust three-step document retrieval process:  
   - **Cosine Similarity**: Quickly fetches the most relevant documents from the **ChromaDB** vector database.  
   - **LLMChainFilter**: Filters out irrelevant documents using the **SmolLMv2-360M-instruct model**, ensuring only relevant content reaches the next stage.  
   - **Cross-Encoder Reranker**: Re-ranks documents for precision using **BAAI/bge-small-en**, fine-tuning results for accuracy.  

2. **Web Search Integration**:  
   Combines knowledge from web searches and local PDFs, allowing it to synthesize rich and well-rounded answers for user queries.

3. **Powered by Cutting-Edge LLMs**:  
   - Utilizes the **Llama-3.2-3B-Instruct-Q8_0.gguf model** to generate detailed and insightful answers.  
   - Efficient inference with Redis-powered batch processing ensures smooth multi-user support.

4. **Optimized for Performance**:  
   - Built with **Langchain** for seamless integration of components.  
   - Uses **Redis** for asynchronous request handling, enabling batch inference and improving responsiveness.  
   - Deployed using **FastAPI** and **Gunicorn**, ensuring scalable and robust performance.

5. **Academic Focus**:  
   Tailored to students and researchers, LAIA simplifies the process of extracting meaningful knowledge from vast resources.

---

## 🛠️ Technologies Used

- **[Langchain](https://github.com/hwchase17/langchain)**: Framework for building LLM applications.  
- **[ChromaDB](https://docs.trychroma.com/)**: Manages and retrieves vectorized document embeddings for RAG.  
- **[Redis](https://redis.io/)**: Enables efficient multi-user support and batch processing.  
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance API framework for serving LAIA.  
- **[Gunicorn](https://gunicorn.org/)**: Python WSGI server for deploying the application.

---

## 💻 Setup and Usage

Follow these simple steps to get started with LAIA:

### 1. Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/LAIA.git
cd LAIA
pip install -r requirements.txt

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

## 🎓 Why Choose LAIA?

- **Comprehensive Knowledge**: Combines the power of web search with textbook-level PDF analysis.  
- **Efficient Multi-User Handling**: Redis ensures smooth and scalable performance, even with multiple simultaneous users.  
- **Advanced Retrieval**: The multi-level filtering in RAG guarantees the highest quality and most relevant information.  
- **Scalability & Performance**: Deployed using Gunicorn and FastAPI for high availability and robust handling of requests.  
- **Designed for Academics**: Whether you're studying for exams or conducting research, LAIA delivers concise and accurate answers.  

---

## 🛠️ Future Roadmap

- **Improved PDF handling**: Improve the way PDFs are being loaded and ingest in the vector database. 
- **Improved Language Model Support**: Add more advanced LLMs for broader language capabilities.  
- **Fine-Tuned Reranking**: Incorporate more efficient cross-encoders for faster processing.  
- **Expanded File Support**: Add support for different file types, including PowerPoint presentations and spreadsheets.  
- **UI Development**: Build a user-friendly web interface to enhance accessibility.  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

Special thanks to the creators of:
- **[Langchain](https://github.com/hwchase17/langchain)** for simplifying LLM integrations.  
- **[ChromaDB](https://docs.trychroma.com/)** for efficient vector storage and retrieval.  
- **[Hugging Face](https://huggingface.co/)** for their open-source models like **BAAI/bge-small-en** and **SmolLMv2-360M-instruct**.

---

## 📩 Contact

For issues, questions, or contributions, feel free to open an issue or contact [sahil.chachra3@live.com](mailto:sahil.chachra3@live.com).
