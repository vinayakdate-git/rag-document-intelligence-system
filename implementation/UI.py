import gradio as gr
import time
import fitz
import shutil
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from openai import OpenAI
from .property_loader_util import get_property_value
from .answer import answer_question
from .injest import create_chunks, append_embeddings

load_dotenv(override=True)

FILE_BASE = Path(__file__).parent.parent
UPLOAD_FOLDER = FILE_BASE / "knowledge-base" / "uploaded_on_UI"
UPLOAD_FOLDER.mkdir(exist_ok=True)

DB_NAME = str(FILE_BASE / "preprocessed_data")
collection_name = "docs"

embedding_model = get_property_value("embedding_model", "model_details.yaml")

openai = OpenAI()

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)


# -------------------------
# Helpers
# -------------------------

def get_sample_questions():
    raw = get_property_value("sample_questions", "example_questions_data.yaml")
    return [q.strip("- ").strip() for q in raw.split("\n") if q.strip()]


def format_context(context):
    result = "### 📚 Retrieved Sources\n\n"

    if not context:
        return "No sources retrieved."

    for i, doc in enumerate(context, start=1):
        snippet = doc.page_content[:350].replace("\n", " ")
        result += f"""
**[{i}] Source:** {doc.metadata['source']}

> {snippet}...

"""
    return result


def get_kb_stats():
    try:
        count = collection.count()
    except:
        count = "Unknown"

    return f"""
### 📊 Knowledge Base

**Indexed Chunks:** {count}  
Vector DB: **ChromaDB**  
Embedding Model: **{get_property_value("embedding_model", "model_details.yaml")}**  
LLM: **{(get_property_value("model_name", "model_details.yaml")).split("/")[-1]}**
"""


# -------------------------
# Chat Streaming
# -------------------------

def chat_stream(message, history):

    start = time.time()

    formatted_history = []
    for q, a in history:
        formatted_history.append({"role": "user", "content": q})
        formatted_history.append({"role": "assistant", "content": a})

    answer, context_data = answer_question(message, formatted_history)

    latency = round(time.time() - start, 2)

    history.append((message, ""))

    partial = ""

    for char in answer:
        partial += char
        history[-1] = (message, partial)

        yield "", history, format_context(context_data), f"""
### ⚙️ System Metrics

Response Time: **{latency} sec**

Chunks Retrieved: **{len(context_data)}**
"""


# -------------------------
# PDF Upload
# -------------------------

def process_uploaded_pdf_end_to_end(file):

    if file is None:
        return "No file uploaded."

    destination = UPLOAD_FOLDER / Path(file.name).name

    if destination.exists():
        return f"⚠️ **{destination.name} already exists in knowledge base.**"

    shutil.copy(file.name, destination)

    pdf_text = ""

    with fitz.open(destination) as pdf:
        for page in pdf:
            pdf_text += page.get_text()

    documents = [
        {
            "type": "uploaded_on_UI",
            "source": str(destination),
            "text": pdf_text
        }
    ]

    chunks = create_chunks(documents)
    append_embeddings(chunks)

    return f"""
✅ **{destination.name} uploaded successfully**

Chunks created: **{len(chunks)}**
"""


def start_processing():
    return gr.update(interactive=False), "⏳ Processing document..."


def finish_processing():
    return gr.update(interactive=True), "✅ Document ready for queries"


# -------------------------
# UI
# -------------------------

def main():

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="blue",
        font=["Inter", "system-ui"]
    )

    with gr.Blocks(title="AI PDF Chatbot", theme=theme) as ui:

        # Header
        gr.Markdown("""
        <div style="text-align:center; font-size:36px; font-weight:bold; color:#ff7a18;">
        🤖 AI-Powered PDF Chatbot
        </div>
        <div style="text-align:center;">
        Interact with PDF documents using <b>natural language queries</b>
        </div>
        <div style="text-align:center; color:#1f77b4;">
        Architecture: RAG + Vector Search + LLM
        </div>
        """)

        with gr.Row():

            # -------- CHAT PANEL --------
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(height="60vh")

                message = gr.Textbox(
                    placeholder="Ask a question...",
                    show_label=False
                )

                # Example Questions
                gr.Markdown("### 💡 Example Questions")

                questions = get_sample_questions()
                buttons = []

                for q in questions:
                    btn = gr.Button(q)
                    buttons.append((btn, q))

            # -------- RIGHT PANEL (TABS = HORIZONTAL NAVIGATION) --------
            with gr.Column(scale=2):

                with gr.Tabs():

                    # -------- TAB 1: KNOWLEDGE BASE --------
                    with gr.Tab("📊 Knowledge Base"):
                        stats = gr.Markdown(get_kb_stats())

                    # -------- TAB 2: RETRIEVED SOURCES --------
                    with gr.Tab("📚 Retrieved Sources"):
                        context = gr.Markdown("No sources yet.")

                    # -------- TAB 3: UPLOAD --------
                    with gr.Tab("📂 Upload PDF"):

                        file_upload = gr.File(file_types=[".pdf"])
                        upload_status = gr.Markdown()
                        processing_status = gr.Markdown()

                        file_upload.upload(
                            start_processing,
                            outputs=[message, processing_status]
                        ).then(
                            process_uploaded_pdf_end_to_end,
                            inputs=file_upload,
                            outputs=upload_status
                        ).then(
                            finish_processing,
                            outputs=[message, processing_status]
                        )

        # Outputs mapping
        chat_outputs = [message, chatbot, context, stats]

        # Bind example buttons
        for btn, q in buttons:
            btn.click(
                fn=lambda x=q: x,
                outputs=message
            ).then(
                chat_stream,
                inputs=[message, chatbot],
                outputs=chat_outputs
            )

        # Submit
        message.submit(
            chat_stream,
            inputs=[message, chatbot],
            outputs=chat_outputs
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()