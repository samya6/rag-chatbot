import gradio as gr

from llm import load_llm
from embeddings import load_embeddings
from rag_pipeline import process_pdf, generate_answer
from utils import get_file_hash


# =========================
# LOAD MODELS
# =========================

llm = load_llm()
embeddings = load_embeddings()

pdf_cache = {}


# =========================
# CHAT FUNCTION (WITH TYPING)
# =========================

def chat_fn(file, message, history):

    if file is None:
        yield history + [[message, "Please upload a PDF first."]]
        return

    if not message.strip():
        yield history
        return

    file_id = get_file_hash(file.name)

    if file_id not in pdf_cache:
        vectorstore, full_text = process_pdf(file.name, embeddings)
        pdf_cache[file_id] = (vectorstore, full_text)

    vectorstore, full_text = pdf_cache[file_id]

    answer, _ = generate_answer(message, vectorstore, full_text, llm)

    # typing animation (word-based)
    partial = ""
    for word in answer.split():
        partial += word + " "
        yield history + [[message, partial]]


# =========================
# PREMIUM DARK UI
# =========================

css = """
html, body, .gradio-container {
    background: linear-gradient(135deg, #0f172a, #020617) !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* Container */
.main-container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
}

/* Title */
.title {
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Chatbox */
.chatbot {
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.15);
}

/* Input */
textarea {
    background: #020617 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Upload */
.upload-box {
    border: 1px dashed #475569;
    border-radius: 12px;
    padding: 12px;
    background: rgba(2, 6, 23, 0.6);
}

/* Buttons */
button {
    background: linear-gradient(90deg, #38bdf8, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 500;
    transition: 0.3s ease;
}

/* File upload label */
[data-testid="file-upload-label"] {
    background: rgba(56, 189, 248, 0.15) ;
    color: #38bdf8 ;
    border: 1px solid rgba(56, 189, 248, 0.5) ;
    border-radius: 8px ;
    padding: 6px 12px ;
    backdrop-filter: blur(6px);
}

/* Optional hover effect */
[data-testid="file-upload-label"]:hover {
    opacity: 0.9;
}

button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.6);
}

/* Upload box inner text */
[data-testid="file-upload"] span {
    color: #000000; 
    font-weight: 500;
}

/* Optional: make it glow slightly */
[data-testid="file-upload"] span {
    text-shadow: 0 0 6px rgba(56, 189, 248, 0.5);
}
"""


# =========================
# UI BUILD
# =========================

with gr.Blocks(theme=gr.themes.Base(), css=css) as app:

    with gr.Column(elem_classes="main-container"):

        gr.Markdown('<div class="title">ResumeAI Pro</div>')
        gr.Markdown('<div class="subtitle">Intelligent Resume Analysis & Career Assistant</div>')

        file_input = gr.File(label="Upload Resume (PDF)", elem_classes="upload-box")

        chatbot = gr.Chatbot(height=520, elem_classes="chatbot")

        msg = gr.Textbox(
            placeholder="Ask anything about your resume...",
            label=""
        )

        clear = gr.Button("Clear Chat")

        msg.submit(
            fn=chat_fn,
            inputs=[file_input, msg, chatbot],
            outputs=chatbot,
            queue=True
        )

        clear.click(lambda: [], None, chatbot)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.launch()