import os
import sys
import sqlite3
from typing import List, Literal, TypedDict, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.tools import TavilySearchResults
from langgraph.graph import END, START, StateGraph

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    sys.stderr.write("Error: GEMINI_API_KEY environment variable is not set.\n")

# ---------------------------------------------------------------------------
# 1. SQLite Database Management for Temporary Chat Storage
# ---------------------------------------------------------------------------
class SQLiteChatHistory:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        sender TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            sys.stderr.write(f"Database Initialization Error: {e}\n")

    def save_message(self, session_id: str, sender: str, message: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chat_history (session_id, sender, message) VALUES (?, ?, ?)",
                    (session_id, sender, message)
                )
                conn.commit()
        except Exception as e:
            sys.stderr.write(f"Error saving message to database: {e}\n")

    def get_history(self, session_id: str, limit: int = 10) -> List[dict]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT sender, message, timestamp FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit)
                )
                rows = cursor.fetchall()
                return [{"sender": r[0], "message": r[1], "timestamp": r[2]} for r in reversed(rows)]
        except Exception as e:
            sys.stderr.write(f"Error reading chat history: {e}\n")
            return []
            
    def get_all_sessions(self) -> List[str]:
        """Fetch all unique session IDs for the sidebar."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT session_id FROM chat_history ORDER BY max(timestamp) DESC")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            sys.stderr.write(f"Error fetching sessions: {e}\n")
            return []

    def clear_session(self, session_id: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
                conn.commit()
        except Exception as e:
            sys.stderr.write(f"Error clearing session chat history: {e}\n")

# Initialize SQLite database
chat_db = SQLiteChatHistory()

# ---------------------------------------------------------------------------
# 2. Embeddings & LLM Setup
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", # Updated to a standard, widely available Gemini version
    temperature=0,
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# Optional Tavily Web Search tool (requires TAVILY_API_KEY in environment)
tavily_search = TavilySearchResults(k=3) if os.getenv("TAVILY_API_KEY") else None

# Session Store for Vector Retrievers
VECTOR_STORES = {}

# ---------------------------------------------------------------------------
# 3. Dynamic PDF Loader & Indexing
# ---------------------------------------------------------------------------
def process_uploaded_pdfs(pdf_paths: List[str], session_id: str):
    try:
        loaded_docs = []
        for path in pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                loaded_docs.extend(loader.load())

        if not loaded_docs:
            return None

        semantic_chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )
        chunks = semantic_chunker.split_documents(loaded_docs)

        # Create new index and save to session store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Important: Allow fetching up to 4 chunks per retrieval
        VECTOR_STORES[session_id] = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Clean up temp file
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)
                
        return True

    except Exception as e:
        sys.stderr.write(f"Error processing PDFs for session {session_id}: {e}\n")
        return False

# ---------------------------------------------------------------------------
# 4. State & Graph Workflow Setup
# ---------------------------------------------------------------------------
class State(TypedDict):
    session_id: str
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(..., description="True if external documents are needed.")

decide_retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return JSON: should_retrieve (boolean). True if answering requires specific facts from uploaded documents."),
    ("human", "Question: {question}"),
])
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

def decide_retrieval(state: State):
    try:
        decision: RetrieveDecision = should_retrieve_llm.invoke(
            decide_retrieval_prompt.format_messages(question=state["question"])
        )
        return {"need_retrieval": decision.should_retrieve}
    except Exception as e:
        sys.stderr.write(f"Error in decide_retrieval: {e}\n")
        return {"need_retrieval": True}

def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"

direct_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using general knowledge or Tavily search context. Be concise and accurate."),
    ("human", "Question: {question}\n\nSearch Context:\n{search_context}"),
])

def generate_direct(state: State):
    search_context = ""
    if tavily_search:
        try:
            results = tavily_search.invoke(state["question"])
            if results:
                search_context = "\n".join([r.get("content", "") for r in results if isinstance(r, dict)])
        except Exception as e:
            sys.stderr.write(f"Tavily search execution error: {e}\n")

    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"],
            search_context=search_context
        )
    )
    return {"answer": out.content}

def retrieve(state: State):
    session_id = state.get("session_id", "default")
    retriever = VECTOR_STORES.get(session_id)
    
    if not retriever:
        return {"docs": []}
    
    q = state.get("retrieval_query") or state["question"]
    try:
        retrieved_docs = retriever.invoke(q)
        return {"docs": retrieved_docs}
    except Exception as e:
        sys.stderr.write(f"Error during retrieval: {e}\n")
        return {"docs": []}

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(..., description="True ONLY if the document contains info related to the question.")

is_relevant_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return JSON: is_relevant. A document is relevant if it discusses the same topic area as the question."),
    ("human", "Question:\n{question}\n\nDocument:\n{document}"),
])
relevance_llm = llm.with_structured_output(RelevanceDecision)

def is_relevant(state: State):
    relevant_docs = []
    for doc in state.get("docs", []):
        try:
            decision = relevance_llm.invoke(
                is_relevant_prompt.format_messages(
                    question=state["question"], 
                    document=doc.page_content
                )
            )
            if decision and decision.is_relevant:
                relevant_docs.append(doc)
        except Exception as e:
            sys.stderr.write(f"Error during relevance checking: {e}\n")

    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"

rag_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant answering based on retrieved document contexts. Answer accurately and directly."),
    ("human", "Question:\n{question}\n\nContext:\n{context}"),
])

def generate_from_context(state: State):
    context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
    if not context:
        return {"answer": "No relevant context found in documents.", "context": ""}
    
    out = llm.invoke(rag_generation_prompt.format_messages(question=state["question"], context=context))
    return {"answer": out.content, "context": context}

def no_answer_found(state: State):
    if tavily_search:
        try:
            results = tavily_search.invoke(state["question"])
            if results:
                web_text = "\n".join([r.get("content", "") for r in results if isinstance(r, dict)])
                out = llm.invoke(direct_generation_prompt.format_messages(
                    question=state["question"], search_context=web_text
                ))
                return {"answer": out.content, "context": "Tavily Web Search"}
        except Exception as e:
            sys.stderr.write(f"Tavily search fallback error: {e}\n")

    return {"answer": "I could not find a relevant answer in the uploaded documents or context.", "context": ""}

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

issup_prompt = ChatPromptTemplate.from_messages([
    ("system", "Verify if the ANSWER is supported by the CONTEXT. Return JSON: issup, evidence."),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"),
])
issup_llm = llm.with_structured_output(IsSUPDecision)

def is_sup(state: State):
    try:
        decision = issup_llm.invoke(
            issup_prompt.format_messages(
                question=state["question"], 
                answer=state.get("answer", ""), 
                context=state.get("context", "")
            )
        )
        return {"issup": decision.issup, "evidence": decision.evidence}
    except Exception as e:
        sys.stderr.write(f"Error in is_sup evaluation: {e}\n")
        return {"issup": "fully_supported", "evidence": []}

def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= 3:
        return "accept_answer"
    return "revise_answer"

revise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a STRICT reviser. Use ONLY the CONTEXT to rewrite the answer using direct facts."),
    ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
])

def revise_answer(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"], 
            answer=state.get("answer", ""), 
            context=state.get("context", "")
        )
    )
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")

isuse_prompt = ChatPromptTemplate.from_messages([
    ("system", "Judge USEFULNESS of the ANSWER for the QUESTION. Return JSON: isuse, reason."),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
])
isuse_llm = llm.with_structured_output(IsUSEDecision)

def is_use(state: State):
    try:
        decision = isuse_llm.invoke(
            isuse_prompt.format_messages(
                question=state["question"], 
                answer=state.get("answer", "")
            )
        )
        return {"isuse": decision.isuse, "use_reason": decision.reason}
    except Exception as e:
        sys.stderr.write(f"Error in is_use evaluation: {e}\n")
        return {"isuse": "useful", "use_reason": ""}

def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= 2:
        return "no_answer_found"
    return "rewrite_question"

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(..., description="Rewritten query optimized for vector retrieval.")

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the QUESTION into a query optimized for vector retrieval. Output JSON: retrieval_query"),
    ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer:\n{answer}"),
])
rewrite_llm = llm.with_structured_output(RewriteDecision)

def rewrite_question(state: State):
    try:
        decision = rewrite_llm.invoke(
            rewrite_prompt.format_messages(
                question=state["question"], 
                retrieval_query=state.get("retrieval_query", ""), 
                answer=state.get("answer", "")
            )
        )
        query = decision.retrieval_query
    except Exception as e:
        sys.stderr.write(f"Error in rewrite_question: {e}\n")
        query = state["question"]

    return {
        "retrieval_query": query, 
        "rewrite_tries": state.get("rewrite_tries", 0) + 1, 
        "docs": [], 
        "relevant_docs": [], 
        "context": ""
    }

# Build LangGraph workflow
g = StateGraph(State)
g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_answer_found", no_answer_found)
g.add_node("is_sup", is_sup)
g.add_node("revise_answer", revise_answer)
g.add_node("is_use", is_use)
g.add_node("rewrite_question", rewrite_question)

g.add_edge(START, "decide_retrieval")
g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
g.add_edge("no_answer_found", END)
g.add_edge("generate_from_context", "is_sup")
g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
g.add_edge("revise_answer", "is_sup")
g.add_conditional_edges("is_use", route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
g.add_edge("rewrite_question", "retrieve")

app = g.compile()

# ---------------------------------------------------------------------------
# 5. Entry point function for Web API / Frontend Integration
# ---------------------------------------------------------------------------
def process_query(user_question: str, session_id: str = "default_session", pdf_paths: Optional[List[str]] = None) -> dict:
    if pdf_paths:
        process_uploaded_pdfs(pdf_paths, session_id)

    chat_db.save_message(session_id, "user", user_question)

    initial_state = {
        "session_id": session_id,
        "question": user_question,
        "retrieval_query": user_question,
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

    try:
        result = app.invoke(initial_state, config={"recursion_limit": 40})
        
        final_answer = result.get("answer", "Sorry, I couldn't generate an answer.")
        if isinstance(final_answer, list) and len(final_answer) > 0 and isinstance(final_answer[0], dict):
            final_answer = final_answer[0].get('text', str(final_answer))

        chat_db.save_message(session_id, "assistant", str(final_answer))

        stats = {
            "retrieved": len(result.get('docs', []) or []),
            "relevant": len(result.get('relevant_docs', []) or []),
            "issup": result.get('issup', 'N/A'),
            "isuse": result.get('isuse', 'N/A'),
            "direct_answer": not result.get('need_retrieval', True)
        }
        return {"answer": final_answer, "stats": stats, "error": None}

    except Exception as e:
        sys.stderr.write(f"Pipeline error: {e}\n")
        return {"answer": None, "stats": None, "error": str(e)}









# import os
# import sys
# import sqlite3
# from typing import List, Literal, TypedDict, Optional, Union
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.tools import TavilySearchResults
# from langgraph.graph import END, START, StateGraph

# load_dotenv()

# if not os.getenv("GEMINI_API_KEY"):
#     sys.stderr.write("Error: GEMINI_API_KEY environment variable is not set.\n")

# # ---------------------------------------------------------------------------
# # 1. SQLite Database Management for Temporary Chat Storage
# # ---------------------------------------------------------------------------
# class SQLiteChatHistory:
#     def __init__(self, db_path: str = "chat_history.db"):
#         self.db_path = db_path
#         self._init_db()

#     def _get_connection(self):
#         return sqlite3.connect(self.db_path)

#     def _init_db(self):
#         try:
#             with self._get_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("""
#                     CREATE TABLE IF NOT EXISTS chat_history (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         session_id TEXT NOT NULL,
#                         sender TEXT NOT NULL,
#                         message TEXT NOT NULL,
#                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#                     )
#                 """)
#                 conn.commit()
#         except Exception as e:
#             sys.stderr.write(f"Database Initialization Error: {e}\n")

#     def save_message(self, session_id: str, sender: str, message: str):
#         try:
#             with self._get_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute(
#                     "INSERT INTO chat_history (session_id, sender, message) VALUES (?, ?, ?)",
#                     (session_id, sender, message)
#                 )
#                 conn.commit()
#         except Exception as e:
#             sys.stderr.write(f"Error saving message to database: {e}\n")

#     def get_history(self, session_id: str, limit: int = 10) -> List[dict]:
#         try:
#             with self._get_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute(
#                     "SELECT sender, message, timestamp FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
#                     (session_id, limit)
#                 )
#                 rows = cursor.fetchall()
#                 return [{"sender": r[0], "message": r[1], "timestamp": r[2]} for r in reversed(rows)]
#         except Exception as e:
#             sys.stderr.write(f"Error reading chat history: {e}\n")
#             return []

#     def clear_session(self, session_id: str):
#         try:
#             with self._get_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
#                 conn.commit()
#         except Exception as e:
#             sys.stderr.write(f"Error clearing session chat history: {e}\n")

# # Initialize SQLite database
# chat_db = SQLiteChatHistory()

# # ---------------------------------------------------------------------------
# # 2. Embeddings & LLM Setup
# # ---------------------------------------------------------------------------
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     temperature=0,
#     google_api_key=os.environ.get("GEMINI_API_KEY")
# )

# # Optional Tavily Web Search tool (requires TAVILY_API_KEY in environment)
# tavily_search = TavilySearchResults(k=3) if os.getenv("TAVILY_API_KEY") else None

# # Session Store for Vector Retrievers
# VECTOR_STORES = {}

# # ---------------------------------------------------------------------------
# # 3. Dynamic PDF Loader & Indexing
# # ---------------------------------------------------------------------------
# def process_uploaded_pdfs(pdf_paths: List[str], session_id: str):
#     """
#     Loads PDF files, chunks them dynamically with SemanticChunker,
#     creates a FAISS vector index, and stores it in memory under session_id.
#     """
#     try:
#         loaded_docs = []
#         for path in pdf_paths:
#             if os.path.exists(path):
#                 loader = PyPDFLoader(path)
#                 loaded_docs.extend(loader.load())

#         if not loaded_docs:
#             return None

#         semantic_chunker = SemanticChunker(
#             embeddings,
#             breakpoint_threshold_type="percentile",
#             breakpoint_threshold_amount=90
#         )
#         chunks = semantic_chunker.split_documents(loaded_docs)

#         vector_store = FAISS.from_documents(chunks, embeddings)
#         VECTOR_STORES[session_id] = vector_store.as_retriever(search_kwargs={"k": 4})
#         return VECTOR_STORES[session_id]

#     except Exception as e:
#         sys.stderr.write(f"Error processing PDFs for session {session_id}: {e}\n")
#         return None

# # ---------------------------------------------------------------------------
# # 4. State & Graph Workflow Setup
# # ---------------------------------------------------------------------------
# class State(TypedDict):
#     session_id: str
#     question: str
#     retrieval_query: str
#     rewrite_tries: int
#     need_retrieval: bool
#     docs: List[Document]
#     relevant_docs: List[Document]
#     context: str
#     answer: str
#     issup: Literal["fully_supported", "partially_supported", "no_support"]
#     evidence: List[str]
#     retries: int
#     isuse: Literal["useful", "not_useful"]
#     use_reason: str

# class RetrieveDecision(BaseModel):
#     should_retrieve: bool = Field(..., description="True if external documents are needed.")

# decide_retrieval_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Return JSON: should_retrieve (boolean). True if answering requires specific facts from uploaded documents."),
#     ("human", "Question: {question}"),
# ])
# should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

# def decide_retrieval(state: State):
#     try:
#         decision: RetrieveDecision = should_retrieve_llm.invoke(
#             decide_retrieval_prompt.format_messages(question=state["question"])
#         )
#         return {"need_retrieval": decision.should_retrieve}
#     except Exception as e:
#         sys.stderr.write(f"Error in decide_retrieval: {e}\n")
#         return {"need_retrieval": True}

# def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
#     return "retrieve" if state["need_retrieval"] else "generate_direct"

# direct_generation_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer using general knowledge or Tavily search context. Be concise and accurate."),
#     ("human", "Question: {question}\n\nSearch Context:\n{search_context}"),
# ])

# def generate_direct(state: State):
#     search_context = ""
#     # Use Tavily for general search capability if configured
#     if tavily_search:
#         try:
#             results = tavily_search.invoke(state["question"])
#             if results:
#                 search_context = "\n".join([r.get("content", "") for r in results if isinstance(r, dict)])
#         except Exception as e:
#             sys.stderr.write(f"Tavily search execution error: {e}\n")

#     out = llm.invoke(
#         direct_generation_prompt.format_messages(
#             question=state["question"],
#             search_context=search_context
#         )
#     )
#     return {"answer": out.content}

# def retrieve(state: State):
#     session_id = state.get("session_id", "default")
#     retriever = VECTOR_STORES.get(session_id)
    
#     if not retriever:
#         return {"docs": []}
    
#     q = state.get("retrieval_query") or state["question"]
#     try:
#         retrieved_docs = retriever.invoke(q)
#         return {"docs": retrieved_docs}
#     except Exception as e:
#         sys.stderr.write(f"Error during retrieval: {e}\n")
#         return {"docs": []}

# class RelevanceDecision(BaseModel):
#     is_relevant: bool = Field(..., description="True ONLY if the document contains info related to the question.")

# is_relevant_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Return JSON: is_relevant. A document is relevant if it discusses the same topic area as the question."),
#     ("human", "Question:\n{question}\n\nDocument:\n{document}"),
# ])
# relevance_llm = llm.with_structured_output(RelevanceDecision)

# def is_relevant(state: State):
#     relevant_docs = []
#     for doc in state.get("docs", []):
#         try:
#             decision = relevance_llm.invoke(
#                 is_relevant_prompt.format_messages(
#                     question=state["question"], 
#                     document=doc.page_content
#                 )
#             )
#             if decision and decision.is_relevant:
#                 relevant_docs.append(doc)
#         except Exception as e:
#             sys.stderr.write(f"Error during relevance checking: {e}\n")

#     return {"relevant_docs": relevant_docs}

# def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
#     if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
#         return "generate_from_context"
#     return "no_answer_found"

# rag_generation_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an assistant answering based on retrieved document contexts. Answer accurately and directly."),
#     ("human", "Question:\n{question}\n\nContext:\n{context}"),
# ])

# def generate_from_context(state: State):
#     context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
#     if not context:
#         return {"answer": "No relevant context found in documents.", "context": ""}
    
#     out = llm.invoke(rag_generation_prompt.format_messages(question=state["question"], context=context))
#     return {"answer": out.content, "context": context}

# def no_answer_found(state: State):
#     # Attempt Tavily search fallback if local dynamic documents yield no answer
#     if tavily_search:
#         try:
#             results = tavily_search.invoke(state["question"])
#             if results:
#                 web_text = "\n".join([r.get("content", "") for r in results if isinstance(r, dict)])
#                 out = llm.invoke(direct_generation_prompt.format_messages(
#                     question=state["question"], search_context=web_text
#                 ))
#                 return {"answer": out.content, "context": "Tavily Web Search"}
#         except Exception as e:
#             sys.stderr.write(f"Tavily search fallback error: {e}\n")

#     return {"answer": "I could not find a relevant answer in the uploaded documents or context.", "context": ""}

# class IsSUPDecision(BaseModel):
#     issup: Literal["fully_supported", "partially_supported", "no_support"]
#     evidence: List[str] = Field(default_factory=list)

# issup_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Verify if the ANSWER is supported by the CONTEXT. Return JSON: issup, evidence."),
#     ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"),
# ])
# issup_llm = llm.with_structured_output(IsSUPDecision)

# def is_sup(state: State):
#     try:
#         decision = issup_llm.invoke(
#             issup_prompt.format_messages(
#                 question=state["question"], 
#                 answer=state.get("answer", ""), 
#                 context=state.get("context", "")
#             )
#         )
#         return {"issup": decision.issup, "evidence": decision.evidence}
#     except Exception as e:
#         sys.stderr.write(f"Error in is_sup evaluation: {e}\n")
#         return {"issup": "fully_supported", "evidence": []}

# def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
#     if state.get("issup") == "fully_supported" or state.get("retries", 0) >= 3:
#         return "accept_answer"
#     return "revise_answer"

# revise_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a STRICT reviser. Use ONLY the CONTEXT to rewrite the answer using direct facts."),
#     ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
# ])

# def revise_answer(state: State):
#     out = llm.invoke(
#         revise_prompt.format_messages(
#             question=state["question"], 
#             answer=state.get("answer", ""), 
#             context=state.get("context", "")
#         )
#     )
#     return {"answer": out.content, "retries": state.get("retries", 0) + 1}

# class IsUSEDecision(BaseModel):
#     isuse: Literal["useful", "not_useful"]
#     reason: str = Field(..., description="Short reason in 1 line.")

# isuse_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Judge USEFULNESS of the ANSWER for the QUESTION. Return JSON: isuse, reason."),
#     ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
# ])
# isuse_llm = llm.with_structured_output(IsUSEDecision)

# def is_use(state: State):
#     try:
#         decision = isuse_llm.invoke(
#             isuse_prompt.format_messages(
#                 question=state["question"], 
#                 answer=state.get("answer", "")
#             )
#         )
#         return {"isuse": decision.isuse, "use_reason": decision.reason}
#     except Exception as e:
#         sys.stderr.write(f"Error in is_use evaluation: {e}\n")
#         return {"isuse": "useful", "use_reason": ""}

# def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
#     if state.get("isuse") == "useful":
#         return "END"
#     if state.get("rewrite_tries", 0) >= 2:
#         return "no_answer_found"
#     return "rewrite_question"

# class RewriteDecision(BaseModel):
#     retrieval_query: str = Field(..., description="Rewritten query optimized for vector retrieval.")

# rewrite_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Rewrite the QUESTION into a query optimized for vector retrieval. Output JSON: retrieval_query"),
#     ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer:\n{answer}"),
# ])
# rewrite_llm = llm.with_structured_output(RewriteDecision)

# def rewrite_question(state: State):
#     try:
#         decision = rewrite_llm.invoke(
#             rewrite_prompt.format_messages(
#                 question=state["question"], 
#                 retrieval_query=state.get("retrieval_query", ""), 
#                 answer=state.get("answer", "")
#             )
#         )
#         query = decision.retrieval_query
#     except Exception as e:
#         sys.stderr.write(f"Error in rewrite_question: {e}\n")
#         query = state["question"]

#     return {
#         "retrieval_query": query, 
#         "rewrite_tries": state.get("rewrite_tries", 0) + 1, 
#         "docs": [], 
#         "relevant_docs": [], 
#         "context": ""
#     }

# # Build LangGraph workflow
# g = StateGraph(State)
# g.add_node("decide_retrieval", decide_retrieval)
# g.add_node("generate_direct", generate_direct)
# g.add_node("retrieve", retrieve)
# g.add_node("is_relevant", is_relevant)
# g.add_node("generate_from_context", generate_from_context)
# g.add_node("no_answer_found", no_answer_found)
# g.add_node("is_sup", is_sup)
# g.add_node("revise_answer", revise_answer)
# g.add_node("is_use", is_use)
# g.add_node("rewrite_question", rewrite_question)

# g.add_edge(START, "decide_retrieval")
# g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
# g.add_edge("generate_direct", END)
# g.add_edge("retrieve", "is_relevant")
# g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
# g.add_edge("no_answer_found", END)
# g.add_edge("generate_from_context", "is_sup")
# g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
# g.add_edge("revise_answer", "is_sup")
# g.add_conditional_edges("is_use", route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
# g.add_edge("rewrite_question", "retrieve")

# app = g.compile()

# # ---------------------------------------------------------------------------
# # 5. Entry point function for Web API / Frontend Integration
# # ---------------------------------------------------------------------------
# def process_query(user_question: str, session_id: str = "default_session", pdf_paths: Optional[List[str]] = None) -> dict:
#     """
#     Main handler called per request. Processes PDFs dynamically per session, 
#     persists history in SQLite, executes Self-RAG logic, and returns output.
#     """
#     # 1. Process PDFs dynamically if uploaded for this session
#     if pdf_paths:
#         process_uploaded_pdfs(pdf_paths, session_id)

#     # 2. Record User Query in SQLite
#     chat_db.save_message(session_id, "user", user_question)

#     initial_state = {
#         "session_id": session_id,
#         "question": user_question,
#         "retrieval_query": user_question,
#         "rewrite_tries": 0,
#         "docs": [],
#         "relevant_docs": [],
#         "context": "",
#         "answer": "",
#         "issup": "no_support",
#         "evidence": [],
#         "retries": 0,
#         "isuse": "not_useful",
#         "use_reason": "",
#     }

#     try:
#         result = app.invoke(initial_state, config={"recursion_limit": 40})
        
#         final_answer = result.get("answer", "Sorry, I couldn't generate an answer.")
#         if isinstance(final_answer, list) and len(final_answer) > 0 and isinstance(final_answer[0], dict):
#             final_answer = final_answer[0].get('text', str(final_answer))

#         # 3. Save Bot Response in SQLite
#         chat_db.save_message(session_id, "bot", str(final_answer))

#         stats = {
#             "retrieved": len(result.get('docs', []) or []),
#             "relevant": len(result.get('relevant_docs', []) or []),
#             "issup": result.get('issup', 'N/A'),
#             "isuse": result.get('isuse', 'N/A'),
#             "direct_answer": not result.get('need_retrieval', True)
#         }
#         return {"answer": final_answer, "stats": stats, "error": None}

#     except Exception as e:
#         sys.stderr.write(f"Pipeline error: {e}\n")
#         return {"answer": None, "stats": None, "error": str(e)}