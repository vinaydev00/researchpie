import streamlit as st
from dotenv import load_dotenv
from app.config import SUPPORTED_EXTENSIONS, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K
from app.history import append_message, clear_history, load_history
from app.rag import answer_question, build_vector_store, clear_vector_store, extract_documents, extract_model_names
import logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_history()
    st.session_state.setdefault("vector_ready", VECTOR_STORE_DIR.exists())
    st.session_state.setdefault("indexed_files", [])
    st.session_state.setdefault("show_sources", True)
    st.session_state.setdefault("top_k", RETRIEVAL_TOP_K)

def render_sidebar(models):
    with st.sidebar:
        st.markdown("## ResearchPie")
        selected_model = st.selectbox("Chat model", options=models, index=0)
        st.session_state["top_k"] = st.slider("Top-K chunks", 1, 10, value=st.session_state["top_k"])
        st.session_state["show_sources"] = st.toggle("Show source citations", value=st.session_state["show_sources"])
        st.divider()
        if st.session_state["vector_ready"]:
            st.success("Knowledge base is active")
            for f in st.session_state["indexed_files"]:
                st.caption(f)
        else:
            st.warning("No knowledge base yet")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear KB", use_container_width=True):
                clear_vector_store()
                st.session_state["vector_ready"] = False
                st.session_state["indexed_files"] = []
                st.rerun()
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                clear_history()
                st.rerun()
    return selected_model

def render_chat_history():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_chat_input(selected_model):
    prompt = st.chat_input("Ask a question about your uploaded documents")
    if not prompt:
        return
    append_message("user", prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not st.session_state["vector_ready"]:
            response = "Please upload documents and click Build knowledge base first."
            st.warning(response)
        else:
            with st.spinner("Searching documents..."):
                try:
                    response = answer_question(question=prompt, model=selected_model)
                    if not st.session_state["show_sources"] and "Sources consulted" in response:
                        response = response.split("---")[0].strip()
                except Exception as exc:
                    response = f"Error: {exc}"
            st.markdown(response)
    append_message("assistant", response)
    st.session_state["messages"].append({"role": "assistant", "content": response})

def render_upload_panel():
    st.subheader("Upload Documents")
    uploads = st.file_uploader("Drop files here", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS], accept_multiple_files=True)
    if st.button("Build Knowledge Base", use_container_width=True, disabled=not uploads, type="primary"):
        try:
            with st.spinner("Reading files and building embeddings..."):
                documents, file_names = extract_documents(uploads or [])
                build_vector_store(documents)
                st.session_state["vector_ready"] = True
                st.session_state["indexed_files"] = file_names
                st.session_state["messages"] = []
                clear_history()
            st.success(f"Knowledge base built from {len(file_names)} file(s).")
        except Exception as exc:
            st.error(f"Failed: {exc}")

def main():
    st.set_page_config(page_title="ResearchPie", page_icon="🥧", layout="wide")
    initialize_state()
    models = extract_model_names()
    selected_model = render_sidebar(models)
    st.title("🥧 ResearchPie")
    st.caption("Upload documents, build a knowledge base, chat with grounded answers.")
    st.divider()
    left_col, right_col = st.columns([1, 1.6], gap="large")
    with left_col:
        render_upload_panel()
    with right_col:
        st.subheader("Chat")
        render_chat_history()
        handle_chat_input(selected_model)

if __name__ == "__main__":
    main()