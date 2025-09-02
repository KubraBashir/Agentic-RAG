# ===============================
# Agentic PDF + Image + DOCX + PPTX Assistant
# ChatGPT-like Streamlit UI (auto-index + full preview + voice)
# ===============================

import os
import base64
from io import BytesIO

import streamlit as st
from streamlit.components.v1 import html as st_html
import warnings
import logging
import sys
import warnings
import os

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Redirect stderr to prevent warnings from appearing on the UI
sys.stderr = open(os.devnull, 'w')


# Suppress all warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress Streamlit's internal warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)


# --- Initialize session state for chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# Voice helpers (browser STT + offline TTS)
try:
    from streamlit_mic_recorder import mic_recorder, speech_to_text
    #from st_mic_recorder import speech_to_text  # browser Web Speech API (Chrome/Edge best)
    MIC_OK = True
except Exception:
    MIC_OK = False

try:
    import pyttsx3  # offline TTS
    TTS_OK = True
except Exception:
    TTS_OK = False

# ========== YOUR ORIGINAL CODE (kept intact) ==========
print("Hello, world!")

# =========================
# 📚 IMPORTS & SETUP
# =========================


import numpy as np
import pandas as pd
import PyPDF2
from PIL import Image
from io import BytesIO
import fitz  # PyMuPDF
import pdfplumber
import cv2
import torch
import faiss
import sympy as sp
import re
from docx2pdf import convert
from pdf2image import convert_from_path


from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from voice import *
import time
import threading
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from instructions import *
from functions import *


# ======== STATE (mirror your globals) ========
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(512)
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = ""
if "ext" not in st.session_state:
    st.session_state.ext = ""
if "last_indexed_key" not in st.session_state:
    st.session_state.last_indexed_key = None
if "speak_responses" not in st.session_state:
    st.session_state.speak_responses = False

memory = st.session_state.memory
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
faiss_index = st.session_state.faiss_index
all_chunks = st.session_state.all_chunks

# =========================
# 🧮 CALCULATOR TOOL
# =========================
def solve_calculation(input_str: str) -> str:
    try:
        result = eval(input_str)
        return f"Result: {result}"
    except Exception:
        try:
            expr = sp.sympify(input_str)
            return f"Result: {expr.evalf()}"
        except Exception as e:
            return f"Error: {str(e)}"

# =========================
# 🔎 EMBEDDINGS + FAISS
# =========================
def generate_text_embedding(text: str) -> np.ndarray:
    emb = embedder.encode(text)
    return np.pad(emb, (0, 128), 'constant')

def generate_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.detach().numpy()[0]

def split_text(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def search_faiss(query: str, top_k=3):
    query_emb = generate_text_embedding(query)
    D, I = faiss_index.search(np.array([query_emb]), top_k)
    return I[0].tolist()

# =========================
# 🛠 AGENT TOOLS
# =========================
def search_knowledge(query: str) -> str:
    idxs = search_faiss(query)
    return "\n\n".join(all_chunks[i] for i in idxs)

def image_query_tool(query: str) -> str:
    ext_local = st.session_state.ext.lower()
    cpath = st.session_state.current_file_path
    if ext_local == 'pdf':
        images = extract_images_from_pdf(cpath)
    elif ext_local == 'docx':
        images = extract_images_from_docx(cpath)
    elif ext_local == 'pptx':
        images = extract_images_from_pptx(cpath)
    else:
        return "(unsupported file type)"
    texts = [extract_text_from_image(img) for img in images]
    return " ".join(texts) or "(no OCR text found)"

def table_query_tool(query: str) -> str:
    ext_local = st.session_state.ext.lower()
    cpath = st.session_state.current_file_path
    if ext_local == 'pdf':
        tables = extract_tables_from_pdf(cpath)
        return "\n\n".join(df.to_string(index=False) for df in tables) or "No tables found."
    elif ext_local == 'docx':
        tables = extract_tables_from_docx(cpath)
    elif ext_local == 'pptx':
        tables = extract_tables_from_pptx(cpath)
    else:
        return "(unsupported file type)"
    table_strings = []
    for table in tables:
        try:
            df = pd.DataFrame(table)
            table_strings.append(df.to_string(index=False))
        except:
            continue
    return "\n\n".join(table_strings) or "No tables found."

def memory_tool(query: str) -> str:
    return "\n".join(f"{m.content}" for m in memory.chat_memory.messages)

def answer_image_tool(input_str):
    try:
        question = input_str.strip()
        cpath = st.session_state.current_file_path
        ext_local = os.path.splitext(cpath)[1].lower()

        if ext_local == '.pdf':
            images = extract_images_from_pdf(cpath)
        elif ext_local == '.docx':
            images = extract_images_from_docx(cpath)
        elif ext_local == '.pptx':
            images = extract_images_from_pptx(cpath)
        elif ext_local in ['.jpg', '.jpeg', '.png']:
            images = [Image.open(cpath).convert("RGB")]
        else:
            return "Unsupported file type for image answering."

        if not images:
            return "No images found in the file."

        responses = []
        for idx, img in enumerate(images):
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            history = memory.load_memory_variables({}).get("chat_history", [])
            history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question} (Image {idx+1})"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            })

            response = llm.invoke(history)
            reply = getattr(response, 'content', 'No response')
            responses.append(f"🖼️ Image {idx+1}: {reply}")

            memory.save_context({"input": f"{question} (Image {idx+1})"}, {"output": reply})

        return "\n\n".join(responses)

    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool("search_knowledge", search_knowledge, "Search indexed PDF,docx,pptx text/images/tables."),
    Tool("table_query_tool", table_query_tool, "Extract tables from PDF/docx/pptx."),
    Tool("image_query_tool", image_query_tool, "Extract OCR text from images in PDF/docx/pptx."),
    Tool("image_ocr", image_ocr, "OCR a standalone image or extract text from images inside PDF/DOCX/PPTX."),
    Tool("memory_tool", memory_tool, "Recall conversation history."),
    Tool("answer_image", answer_image_tool, "Ask about an image/pdf/docx/pptx: 'image_path||your question'."),
    Tool("solve_calculation", solve_calculation, "Solve math expressions.")
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def agent_run(query: str) -> str:
    response = agent.run(query)
    return response

# =========================
# 🎨 UI (auto-index + full preview + voice)
# =========================
st.set_page_config(page_title="Agentic Assistant", page_icon="🤖", layout="wide")

# —— CSS (tidy bubbles + fixed input + scrollable chat list) ——

st.markdown("""
<style>
/* 📌 Make the right (preview) column scrollable inside a fixed height */
.element-container:has(> .stMarkdown + .stImage),
.element-container:has(> .stMarkdown + div[data-testid="stVerticalBlock"]) {
    max-height: 85vh;
    overflow-y: auto;
    padding-right: 10px;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
}

/* Optional: style scrollbar */
.element-container::-webkit-scrollbar {
    width: 10px;
}
.element-container::-webkit-scrollbar-thumb {
    background-color: #aaa;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# layout: left chat / right preview
left, right = st.columns([0.62, 0.38], gap="large")

with st.sidebar:
    st.header("📎 Upload")
    uploaded = st.file_uploader(
        "Upload PDF/DOCX/PPTX/Image",
        type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"]
    )
    st.checkbox("🔊 Speak responses", key="speak_responses")
    if st.button("Clear", use_container_width=True):
        faiss_index.reset()
        all_chunks.clear()
        memory.clear()
        memory.chat_memory.messages = []
        st.session_state.current_file_path = ""
        st.session_state.ext = ""
        st.session_state.last_indexed_key = None
        st.success("Cleared memory & embeddings.")

def save_upload(file):
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    delete_file_later(path, delay=200)  # ⏱️ delete after 10 minutes (200 seconds)
    return path


def delete_file_later(path, delay=200):
    def delayed_delete():
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
    threading.Thread(target=delayed_delete).start()


# ===== auto-index exactly like your CLI =====
def index_current_file(file_path, ext_local):
    faiss_index.reset()
    all_chunks.clear()
    memory.clear()
    memory.chat_memory.messages = []
    st.toast("🧹 Cleared previous memory & embeddings.")

    if ext_local == 'pdf':
        text = extract_text_from_pdf(file_path)
        for chunk in split_text(text):
            faiss_index.add(np.array([generate_text_embedding(chunk)]))
            all_chunks.append(f"[TEXT] {chunk}")

        for img in extract_images_from_pdf(file_path):
            ocr = extract_text_from_image(img)
            for och in split_text(ocr):
                faiss_index.add(np.array([generate_text_embedding(och)]))
                all_chunks.append(f"[IMAGE_OCR] {och}")
            faiss_index.add(np.array([generate_image_embedding(img)]))
            all_chunks.append("[IMAGE_EMB] vector")

        for df in extract_tables_from_pdf(file_path):
            try:
                if isinstance(df, pd.DataFrame):
                    table_text = df.to_string(index=False)
                    for tchunk in split_text(table_text):
                        faiss_index.add(np.array([generate_text_embedding(tchunk)]))
                        all_chunks.append(f"[TABLE] {tchunk}")
            except Exception as e:
                st.write(f"⚠️ Skipping table due to error: {e}")

    elif ext_local == 'docx':
        text = extract_text_from_docx(file_path)
        for chunk in split_text(text):
            faiss_index.add(np.array([generate_text_embedding(chunk)]))
            all_chunks.append(f"[TEXT] {chunk}")

        for img in extract_images_from_docx(file_path):
            ocr = extract_text_from_image(img)
            for och in split_text(ocr):
                faiss_index.add(np.array([generate_text_embedding(och)]))
                all_chunks.append(f"[IMAGE_OCR] {och}")
            faiss_index.add(np.array([generate_image_embedding(img)]))
            all_chunks.append("[IMAGE_EMB] vector")

        tables = extract_tables_from_docx(file_path)
        for table in tables:
            try:
                if isinstance(table, (list, tuple)):
                    df = pd.DataFrame(table)
                    table_text = df.to_string(index=False)
                elif isinstance(table, str):
                    table_text = table
                else:
                    continue
                for tchunk in split_text(table_text):
                    faiss_index.add(np.array([generate_text_embedding(tchunk)]))
                    all_chunks.append(f"[TABLE] {tchunk}")
            except Exception as e:
                st.write(f"⚠️ Skipping DOCX table due to error: {e}")

    elif ext_local == 'pptx':
        text = extract_text_from_pptx(file_path)
        for chunk in split_text(text):
            faiss_index.add(np.array([generate_text_embedding(chunk)]))
            all_chunks.append(f"[TEXT] {chunk}")

        for img in extract_images_from_pptx(file_path):
            ocr = extract_text_from_image(img)
            for och in split_text(ocr):
                faiss_index.add(np.array([generate_text_embedding(och)]))
                all_chunks.append(f"[IMAGE_OCR] {och}")
            faiss_index.add(np.array([generate_image_embedding(img)]))
            all_chunks.append("[IMAGE_EMB] vector")

        tables = extract_tables_from_pptx(file_path)
        for table in tables:
            try:
                if isinstance(table, (list, tuple)):
                    df = pd.DataFrame(table)
                    table_text = df.to_string(index=False)
                elif isinstance(table, str):
                    table_text = table
                else:
                    continue
                for tchunk in split_text(table_text):
                    faiss_index.add(np.array([generate_text_embedding(tchunk)]))
                    all_chunks.append(f"[TABLE] {tchunk}")
            except Exception as e:
                st.write(f"⚠️ Skipping PPTX table due to error: {e}")

# Save + auto-index once
if uploaded:
    file_path = save_upload(uploaded)
    st.session_state.current_file_path = file_path
    st.session_state.ext = file_path.lower().split('.')[-1]
    key = (uploaded.name, uploaded.size)
    if st.session_state.last_indexed_key != key:
        with st.spinner(f"Indexing {st.session_state.ext.upper()}..."):
            index_current_file(file_path, st.session_state.ext)
        st.session_state.last_indexed_key = key
        st.success(f"{st.session_state.ext.upper()} indexed — ready to chat!")

# ---------- Right panel: FULL preview (PDF = all pages, scrollable) ----------
import tempfile

def render_pdf_scrollable(path: str, height_px: int = 820, zoom: float = 1.25):
    try:
        doc = fitz.open(path)
        mat = fitz.Matrix(zoom, zoom)

        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # Create a temporary image file
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            with open(temp_path, "wb") as f:
                f.write(img_bytes)

            # Display the image
            st.image(temp_path, use_container_width=True, caption=f"Page {i}")

            # Auto-delete this preview after 200 seconds
            delete_file_later(temp_path, delay=200)

    except Exception as e:
        st.info(f"Preview unavailable: {e}")

        
def render_docx_as_images(docx_path: str):
    try:
        import pythoncom
        pythoncom.CoInitialize()  # ✅ Initialize COM before using Word automation

        from docx2pdf import convert
        from pdf2image import convert_from_path

        # Convert DOCX to PDF
        pdf_path = docx_path.replace(".docx", "_preview.pdf")
        convert(docx_path, pdf_path)
        delete_file_later(pdf_path, delay=200)


        # Poppler path (✅ your actual path here)
        poppler_path = r"C:\Users\user\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=150, poppler_path=poppler_path)
        for i, img in enumerate(images, 1):
            st.image(img, use_container_width=True, caption=f"Page {i}")

    except Exception as e:
        st.info(f"Preview unavailable: {e}")



with right:
    with st.container():
        st.markdown('<div id="preview-scroll">', unsafe_allow_html=True)

        cpath = st.session_state.current_file_path
        ext_local = st.session_state.ext.lower() if st.session_state.ext else ""
        st.subheader("Preview")
        if cpath:
            st.caption(os.path.basename(cpath))
            try:
                if ext_local == "pdf":
                    render_pdf_scrollable(cpath, height_px=820)
                elif ext_local in ["jpg", "jpeg", "png"]:
                    st.image(cpath, use_container_width=True)
                elif ext_local == "docx":
                    render_docx_as_images(cpath)
                elif ext_local == "pptx":
                    imgs = extract_images_from_pptx(cpath)
                    if imgs:
                        for i, im in enumerate(imgs, start=1):
                            st.image(im, use_container_width=True, caption=f"Slide image {i}")
                    else:
                        st.write("PPTX loaded. No embedded images to preview.")
            except Exception as e:
                st.info(f"Preview unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)



# ---------- Left panel: Chat like ChatGPT (+ voice) ----------
# --- Combined Chat Input (Text + Audio) ---
with left:
    st.markdown("### 💬 Ask your question")

    # --- Display all previous messages ---
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Your existing input code (unchanged) ---
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        typed_prompt = st.chat_input("Type your message here...")

    with col2:
        spoken_text = None
        if MIC_OK:
            spoken_text = speech_to_text(language='en', use_container_width=True,
                                         start_prompt="🎤 Start", stop_prompt="⏹ Stop")
        else:
            st.caption("🎙️ Mic not installed")

    prompt = typed_prompt if typed_prompt else (spoken_text if spoken_text and spoken_text.strip() else None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ext_local = st.session_state.get("ext", "")
        cpath = st.session_state.get("current_file_path", "")
        try:
            query_to_run = f"{cpath}||{prompt}" if ext_local and ext_local not in ['pdf','docx','pptx'] and cpath else prompt
            response = agent_run(query_to_run)
        except Exception as e:
            response = f"⚠️ Agent error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)  
