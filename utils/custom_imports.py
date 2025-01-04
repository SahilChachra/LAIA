import os
import json
import re
import requests
from typing import List, Dict, Any, Optional
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from pydantic import BaseModel, Field, validator
from langchain_community.document_loaders import ArxivLoader
from pathlib import Path
import sys,os,os.path
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader