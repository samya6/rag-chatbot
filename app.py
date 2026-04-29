print("App starting...")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import gradio as gr

# Load PDF
print("Loading PDF...")
loader = PyPDFLoader("Sample.pdf")
documents = loader.load()
print("PDF loaded")

# Chunking
print("Chunking...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)
print("Chunking done")

# Embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embeddings ready")

# Vector store
print("Building vector store...")
vectorstore = FAISS.from_documents(docs, embeddings)
print("Vector store ready")

# LLM
print("Loading model...")
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.5
)
print("Model loaded")

llm = HuggingFacePipeline(pipeline=pipe)

# RAG chain
print("Creating RAG chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
print("RAG ready")

# Function
def ask_question(query):
    return qa_chain.run(query)

# UI
print("Launching UI...")
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG PDF Chatbot"
)

interface.launch()