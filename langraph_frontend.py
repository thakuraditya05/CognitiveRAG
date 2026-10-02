import streamlit as st
import uuid
import os

import tempfile
# =========================== Backend Binding ===========================

# =========================== Backend Binding ===========================
# Properly import the compiled app and database from your backend
from langraph_backend import app as chatbot, chat_db, process_uploaded_pdfs

st.set_page_config(page_title="Self-RAG Intelligence Hub", layout="wide")

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    """Loads chat history directly from your SQLite DB."""
    history = chat_db.get_history(thread_id, limit=50)
    
    temp_messages = []
    for msg in history:
        role = "user" if msg["sender"] == "user" else "assistant"
        temp_messages.append({"role": role, "content": msg["message"]})
        
    return temp_messages

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = chat_db.get_all_sessions()
        
    if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
        add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("🧠 Self-RAG Agent")
st.sidebar.markdown("Equipped with Vector Retrieval, Grading, and Web Search Fallback.")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF for RAG", type=["pdf"])

if uploaded_file:
    with st.spinner("Chunking & Vectorizing PDF..."):
        # Save uploaded file temporarily so the backend PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Call the backend function to process and index the PDF
        process_uploaded_pdfs([tmp_file_path], st.session_state["thread_id"])
        st.sidebar.success("✅ PDF Indexed Successfully!")

st.sidebar.header("History")
for thread_id in st.session_state["chat_threads"]:
    if st.sidebar.button(f"Session: {str(thread_id)[:8]}...", key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.rerun()

# ============================ Main UI ============================
st.title("Self-RAG Intelligence Hub 🚀")
st.caption("Ask me anything! I will retrieve documents, grade their relevance, check for hallucinations, and rewrite queries if necessary.")

# 1st Dropdown: Self-RAG Tool/Node Information
with st.expander("🛠️️ **Self-RAG Node Explanations (Click to expand)**", expanded=False):
    st.markdown("""
    * **`decide_retrieval`**: Router that decides if the query needs local vector retrieval or a direct LLM response.
    * **`retrieve`**: Fetches semantically similar chunks from the FAISS Vector Database.
    * **`is_relevant`**: Grader that evaluates if retrieved chunks actually contain information to answer the question.
    * **`generate_from_context`**: The generator that drafts an answer based *only* on relevant chunks.
    * **`no_answer_found`**: Fallback mechanism that uses Tavily Web Search if local docs fail.
    * **`is_sup` (Hallucination Checker)**: Verifies that the generated answer is completely supported by the retrieved context.
    * **`revise_answer`**: If hallucination is detected, this node forces the LLM to rewrite the answer using strict facts.
    * **`is_use` (Usefulness Checker)**: Evaluates if the final answer actually addresses the user's original query.
    * **`rewrite_question`**: If the answer isn't useful, it rewrites the query for better search results and loops back to retrieval.
    """)

# 2nd Dropdown: Dynamic Node Graph Visualization
with st.expander("🕸️ **Self-RAG Architecture Graph**", expanded=False):
    st.caption("This is the actual LangGraph architecture running under the hood.")
    try:
        # Define the configuration dictionary to customize theme colors safely
        config = {
            "config": {
                "theme": "base",
                "themeVariables": {
                    "primaryTextColor": "#000000",
                    "primaryColor": "#E6E6FA",
                    "edgeLabelBackground": "#333333",
                    "tertiaryTextColor": "#000000"
                }
            }
        }
        
        # Pass the config directly into LangGraph's mermaid generator
        mermaid_code = chatbot.get_graph().draw_mermaid(frontmatter_config=config)
        
        # Render using Streamlit's native Mermaid support
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
    except Exception as e:
        st.warning("Could not render the graph visually. Please ensure you are using Streamlit v1.35+.")
        st.code(str(e))
    

# 1. Render existing history correctly
for message in st.session_state["message_history"]:
    if message["role"] == "tool":
        with st.expander(f"🛠 Node Executed: `{message.get('name', 'Unknown')}`", expanded=False):
            st.code(message["content"], language="json")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 2. Handle new user input
user_input = st.chat_input("Enter your query...")

if user_input:
    # Update UI & Save User message to Database immediately
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    chat_db.save_message(st.session_state["thread_id"], "user", user_input)
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Initial state required by your specific backend TypedDict
    initial_state = {
        "session_id": st.session_state["thread_id"],
        "question": user_input,
        "retrieval_query": user_input,
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
    }

    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        
        # Stream from your custom graph
        events = chatbot.stream(initial_state, config={"recursion_limit": 40})
        
        final_ai_text = ""
        
        for event in events:
            # LangGraph outputs dicts of {node_name: {state_updates}}
            for node_name, node_state in event.items():
                
                # Save raw state text format to history instead of raw dict/json
                summary_parts = [f"{k}: {v}" for k, v in node_state.items() if k not in ["docs", "relevant_docs"]]
                text_content = " | ".join(summary_parts)
                
                st.session_state["message_history"].append({
                    "role": "tool", 
                    "name": node_name, 
                    "content": text_content
                })
                
                # Print as clean text lines inside the expander, never as a dict or JSON
                with st.expander(f"🛠️ Node Executed: `{node_name}`", expanded=True):
                    for k, v in node_state.items():
                        if k not in ["docs", "relevant_docs"]:
                            # If a value is a list or block, format it neatly as text
                            if isinstance(v, list):
                                st.markdown(f"**{k}**:")
                                for item in v:
                                    st.markdown(f"- {item}")
                            else:
                                st.markdown(f"**{k}**: {v}")
                
                # If this node produced an answer, update the placeholder
                if "answer" in node_state and node_state["answer"]:
                    # Handle cases where answer might be a structured block or string
                    ans = node_state["answer"]
                    if isinstance(ans, list):
                        final_ai_text = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in ans])
                    else:
                        final_ai_text = str(ans)
                    st_placeholder.markdown(final_ai_text)
                    
        if final_ai_text:
            st.session_state["message_history"].append({"role": "assistant", "content": final_ai_text})
            chat_db.save_message(st.session_state["thread_id"], "assistant", final_ai_text)