import os
import sys
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from langchain_experimental.text_splitter import SemanticChunker



# Load environment variables
load_dotenv()

# Verify API key presence
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Error: GEMINI_API_KEY environment variable is not set.")
    print("Please set GEMINI_API_KEY in your .env file or environment (get it from Google AI Studio).")
    sys.exit(1)


# -----------------------------
# 0) Load Documents (with Fallback)
# -----------------------------
def load_company_documents() -> List[Document]:
    pdf_paths = [
        "./documents/Company_Policies.pdf",
        "./documents/Company_Profile.pdf",
        "./documents/Product_and_Pricing.pdf",
    ]

    loaded_docs = []
    missing_files = []

    for path in pdf_paths:
        if os.path.exists(path):
            try:
                loaded_docs.extend(PyPDFLoader(path).load())
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
        else:
            missing_files.append(path)

    if loaded_docs:
        print(f"✅ Successfully loaded {len(loaded_docs)} document pages from PDF files.")
        return loaded_docs

    # Fallback mock documents if PDF files do not exist locally
    print(f"⚠️ Local PDF files not found: {missing_files}")
    print("👉 Using in-memory fallback documents for testing...\n")
    return [
        Document(
            page_content=(
                "NexaAI Company Profile: NexaAI is a cutting-edge artificial intelligence "
                "company focusing on enterprise automation. Our company culture values "
                "remote-first work, continuous learning, and transparent communication. "
                "We provide flexible working hours and an annual development budget."
            ),
            metadata={"source": "Company_Profile.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "NexaAI Company Policies: The standard probation period is 3 months. "
                "Employees receive 20 days of paid annual vacation leave and full health benefits."
            ),
            metadata={"source": "Company_Policies.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "NexaAI Product and Pricing: Pro Plan costs $49/month with full features. "
                "Includes a 14-day free trial for new enterprise users."
            ),
            metadata={"source": "Product_and_Pricing.pdf", "page": 1},
        ),
    ]


# -----------------------------
# LLM & Embeddings Setup (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0,
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-2", 
#     google_api_key=os.environ.get("GEMINI_API_KEY")
# )
# -----------------------------

docs = load_company_documents()

# chunks = RecursiveCharacterTextSplitter(
#     chunk_size=600, chunk_overlap=150
# ).split_documents(docs)

# 2. Configure the Semantic Chunker
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile", 
    breakpoint_threshold_amount=90 
)
chunks = semantic_chunker.split_documents(docs)


# Using Google's latest text embedding model  
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Using Gemini 1.5 Flash (free tier eligible, fast, supports structured outputs)
# -----------------------------
# Graph State
# -----------------------------
class State(TypedDict):
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


# -----------------------------
# 1) Decide retrieval
# -----------------------------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False.",
    )


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from company documents.\n"
            "- should_retrieve=False for general explanations/definitions.\n"
            "- If unsure, choose True.",
        ),
        ("human", "Question: {question}"),
    ]
)

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)


def decide_retrieval(state: State):
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


# -----------------------------
# 2) Direct answer (no retrieval)
# -----------------------------
direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only your general knowledge.\n"
            "If it requires specific company info, say:\n"
            "'I don't know based on my general knowledge.'",
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: State):
    out = llm.invoke(
        direct_generation_prompt.format_messages(question=state["question"])
    )
    return {"answer": out.content}


# -----------------------------
# 3) Retrieve
# -----------------------------
def retrieve(state: State):
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}


# -----------------------------
# 4) Relevance filter
# -----------------------------
class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that can directly answer the question.",
    )


is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "When unsure, return is_relevant=true.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)

relevance_llm = llm.with_structured_output(RelevanceDecision)


def is_relevant(state: State):
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def route_after_relevance(
    state: State,
) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


# -----------------------------
# 5) Generate from context
# -----------------------------
rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business rag chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "Task:\n"
            "Answer the question based on the context.\n"
            "Don't mention that you are getting a context in your answer.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


def generate_from_context(state: State):
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()
    if not context:
        return {"answer": "No answer found.", "context": ""}
    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"], context=context
        )
    )
    return {"answer": out.content, "context": context}


def no_answer_found(state: State):
    return {"answer": "No answer found.", "context": ""}


# -----------------------------
# 6) IsSUP verify + revise loop
# -----------------------------
class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n",
        ),
    ]
)

issup_llm = llm.with_structured_output(IsSUPDecision)


def is_sup(state: State):
    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


MAX_RETRIES = 10


def route_after_issup(
    state: State,
) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"


revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "Use ONLY the CONTEXT to rewrite the answer using direct quotes or facts.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}",
        ),
    ]
)


def revise_answer(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }


# -----------------------------
# 7) IsUSE verify + rewrite loop
# -----------------------------
class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n"
            "Return JSON with keys: isuse, reason.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)

isuse_llm = llm.with_structured_output(IsUSEDecision)


def is_use(state: State):
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}


MAX_REWRITE_TRIES = 3


def route_after_isuse(
    state: State,
) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval.",
    )


rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over internal company PDFs.\n"
            "Output JSON with key: retrieval_query",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}",
        ),
    ]
)

rewrite_llm = llm.with_structured_output(RewriteDecision)


def rewrite_question(state: State):
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


# -----------------------------
# Build graph
# -----------------------------
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

g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {"generate_direct": "generate_direct", "retrieve": "retrieve"},
)

g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")

g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found",
    },
)

g.add_edge("no_answer_found", END)
g.add_edge("generate_from_context", "is_sup")

g.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "accept_answer": "is_use",
        "revise_answer": "revise_answer",
    },
)

g.add_edge("revise_answer", "is_sup")

g.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",
    },
)

g.add_edge("rewrite_question", "retrieve")

app = g.compile()


# -----------------------------
# Run execution check
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 Self-RAG Chatbot is Ready!")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    # Start an infinite loop to keep asking questions
    while True:
        user_question = input("\n🧑 You: ")
        
        # Exit condition
        if user_question.lower() in ['quit', 'exit']:
            print("Goodbye! 👋")
            break
            
        # Skip empty questions
        if not user_question.strip():
            continue
            
        print("⏳ Thinking and searching documents...")
        
        # Create the state using your custom question
        initial_state = {
            "question": user_question,
            "retrieval_query": user_question,  # Initial search uses the question
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
            # Run the workflow using your exact config
            result = app.invoke(initial_state, config={"recursion_limit": 80})
            
            # Your code stores the output in "answer"
            final_answer = result.get("answer", "Sorry, I couldn't generate an answer.")
            
            # Clean up the format if it returns as a list/dictionary (like in your previous logs)
            if isinstance(final_answer, list) and len(final_answer) > 0 and isinstance(final_answer[0], dict):
                final_answer = final_answer[0].get('text', str(final_answer))
                
            print("\n🤖 Answer:", final_answer)
            
            # Optional: Print out the background stats so you still see how it performed
            print("\n[Debug Stats]:")
            print(f"  - Retrieved: {len(result.get('docs', []) or [])} | Relevant: {len(result.get('relevant_docs', []) or [])}")
            print(f"  - Is Supported: {result.get('issup')} | Is Useful: {result.get('isuse')}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("\n========================================")
            print("   ❌ WORKFLOW EXECUTION FAILED         ")
            print("========================================")
            print("Error details:", str(e))
            sys.exit(1)