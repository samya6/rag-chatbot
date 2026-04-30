import gradio as gr

from llm import load_llm
from embeddings import load_embeddings
from rag_pipeline import process_pdf, generate_answer
from utils import get_file_hash


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

llm = load_llm()              # LLaMA 3 via Groq
embeddings = load_embeddings()

print("Models loaded successfully")


# =========================
# CACHE
# =========================

pdf_cache = {}


# =========================
# CHAT FUNCTION
# =========================

def chat_fn(file, message, history):
    if file is None:
        return history + [[message, "Please upload a PDF first."]]

    if not message.strip():
        return history

    try:
        file_path = file if isinstance(file, str) else file.name
        file_id = get_file_hash(file_path)

        # Process PDF once
        if file_id not in pdf_cache:
            print("Processing PDF...")
            vectorstore, full_text = process_pdf(file_path, embeddings)
            pdf_cache[file_id] = (vectorstore, full_text)
            print("PDF cached")

        vectorstore, full_text = pdf_cache[file_id]

        print(f"User question: {message}")

        # Generate answer (IMPORTANT: works with ChatGroq)
        answer, docs = generate_answer(message, vectorstore, full_text, llm)

        return history + [[message, answer.strip()]]

    except Exception as e:
        return history + [[message, f"Error: {str(e)}"]]


# =========================
# UI (CHAT STYLE)
# =========================

with gr.Blocks() as app:

    gr.Markdown("# 📄 RAG PDF Chatbot (LLaMA 3 Powered)")
    gr.Markdown("Upload a PDF and ask questions about it.")

    file_input = gr.File(label="Upload PDF")

    chatbot = gr.Chatbot(height=450)

    msg = gr.Textbox(
        placeholder="Ask a question about your document...",
        label="Your Question"
    )

    clear = gr.Button("Clear Chat")

    msg.submit(
        fn=chat_fn,
        inputs=[file_input, msg, chatbot],
        outputs=chatbot
    )

    clear.click(lambda: [], None, chatbot)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.launch()