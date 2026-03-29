# """
# NUST Admissions Chatbot
# ========================
# A fully local, offline RAG-powered chatbot for NUST admission guidance.
# Uses Ollama (LLM + embeddings) + ChromaDB (vector store) + Streamlit (UI).

# Run:
#     streamlit run chatbot.py
# """

# import streamlit as st
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain_community.vectorstores import Chroma
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
# import os
# import time

# # ─────────────────────────────────────────────
# #  CONFIG
# # ─────────────────────────────────────────────
# DB_DIR      = "./nust_db"
# EMBED_MODEL = "nomic-embed-text"
# LLM_MODEL   = "llama3.1:8b"       # Change to mistral:7b if preferred
# TEMPERATURE = 0.1                  # Low = more factual, less creative
# TOP_K_DOCS  = 5                    # How many chunks to retrieve per question

# # ─────────────────────────────────────────────
# #  PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="NUST Admissions Assistant",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# #  CUSTOM CSS
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

#     /* Root theme */
#     :root {
#         --nust-green:    #006633;
#         --nust-gold:     #C9A84C;
#         --nust-dark:     #0a1628;
#         --nust-mid:      #112240;
#         --nust-light:    #1d3461;
#         --text-primary:  #e8edf5;
#         --text-muted:    #8896a8;
#         --accent:        #4fc3f7;
#     }

#     /* Global background */
#     .stApp {
#         background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
#         font-family: 'DM Sans', sans-serif;
#     }

#     /* Hide default streamlit elements */
#     #MainMenu, footer, header { visibility: hidden; }
#     .stDeployButton { display: none; }

#     /* Sidebar */
#     section[data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%);
#         border-right: 1px solid rgba(201,168,76,0.2);
#     }

#     /* Chat messages */
#     .stChatMessage {
#         background: transparent !important;
#         border: none !important;
#     }

#     /* User message bubble */
#     [data-testid="stChatMessageContent"]:has(.user-bubble) {
#         background: transparent;
#     }

#     .user-bubble {
#         background: linear-gradient(135deg, #006633, #004d26);
#         border: 1px solid rgba(201,168,76,0.3);
#         border-radius: 18px 18px 4px 18px;
#         padding: 14px 18px;
#         color: #e8edf5;
#         font-family: 'DM Sans', sans-serif;
#         font-size: 15px;
#         line-height: 1.6;
#         max-width: 85%;
#         margin-left: auto;
#         box-shadow: 0 4px 20px rgba(0,102,51,0.3);
#     }

#     .assistant-bubble {
#         background: linear-gradient(135deg, #112240, #1a2d50);
#         border: 1px solid rgba(79,195,247,0.2);
#         border-radius: 18px 18px 18px 4px;
#         padding: 14px 18px;
#         color: #e8edf5;
#         font-family: 'DM Sans', sans-serif;
#         font-size: 15px;
#         line-height: 1.6;
#         max-width: 90%;
#         box-shadow: 0 4px 20px rgba(0,0,0,0.3);
#     }

#     .assistant-bubble strong { color: #C9A84C; }
#     .assistant-bubble ul { padding-left: 20px; }
#     .assistant-bubble li { margin: 6px 0; }

#     /* Source tags */
#     .source-tag {
#         display: inline-block;
#         background: rgba(201,168,76,0.1);
#         border: 1px solid rgba(201,168,76,0.3);
#         color: #C9A84C;
#         font-size: 11px;
#         padding: 2px 8px;
#         border-radius: 20px;
#         margin: 4px 3px 0 0;
#         font-family: 'DM Sans', sans-serif;
#     }

#     /* Chat input */
#     .stChatInputContainer {
#         background: rgba(17,34,64,0.8) !important;
#         border: 1px solid rgba(201,168,76,0.3) !important;
#         border-radius: 16px !important;
#         padding: 4px !important;
#     }
#     .stChatInput textarea {
#         background: transparent !important;
#         color: #e8edf5 !important;
#         font-family: 'DM Sans', sans-serif !important;
#     }

#     /* Sidebar elements */
#     .sidebar-header {
#         font-family: 'Syne', sans-serif;
#         font-weight: 800;
#         font-size: 22px;
#         color: #C9A84C;
#         letter-spacing: -0.5px;
#         margin-bottom: 4px;
#     }
#     .sidebar-sub {
#         font-size: 13px;
#         color: #8896a8;
#         margin-bottom: 20px;
#     }

#     /* Quick question pills */
#     .stButton > button {
#         background: rgba(17,34,64,0.8) !important;
#         border: 1px solid rgba(79,195,247,0.2) !important;
#         color: #8896a8 !important;
#         border-radius: 20px !important;
#         font-size: 12px !important;
#         font-family: 'DM Sans', sans-serif !important;
#         padding: 6px 14px !important;
#         text-align: left !important;
#         width: 100% !important;
#         transition: all 0.2s !important;
#     }
#     .stButton > button:hover {
#         border-color: rgba(201,168,76,0.5) !important;
#         color: #C9A84C !important;
#         background: rgba(201,168,76,0.05) !important;
#     }

#     /* Main header area */
#     .main-header {
#         text-align: center;
#         padding: 40px 20px 20px;
#     }
#     .main-title {
#         font-family: 'Syne', sans-serif;
#         font-weight: 800;
#         font-size: 42px;
#         background: linear-gradient(135deg, #ffffff 0%, #C9A84C 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         letter-spacing: -1px;
#         line-height: 1.1;
#         margin-bottom: 10px;
#     }
#     .main-subtitle {
#         font-family: 'DM Sans', sans-serif;
#         color: #8896a8;
#         font-size: 16px;
#         font-weight: 300;
#     }

#     /* Status badge */
#     .status-badge {
#         display: inline-flex;
#         align-items: center;
#         gap: 6px;
#         background: rgba(0,102,51,0.15);
#         border: 1px solid rgba(0,102,51,0.4);
#         border-radius: 20px;
#         padding: 4px 12px;
#         font-size: 12px;
#         color: #4caf82;
#         font-family: 'DM Sans', sans-serif;
#         margin-top: 10px;
#     }
#     .status-dot {
#         width: 6px; height: 6px;
#         background: #4caf82;
#         border-radius: 50%;
#         animation: pulse 2s infinite;
#     }
#     @keyframes pulse {
#         0%, 100% { opacity: 1; }
#         50% { opacity: 0.4; }
#     }

#     /* Thinking indicator */
#     .thinking {
#         color: #8896a8;
#         font-style: italic;
#         font-size: 13px;
#         padding: 10px 0;
#     }

#     /* Divider */
#     hr { border-color: rgba(201,168,76,0.15) !important; }

#     /* Scrollbar */
#     ::-webkit-scrollbar { width: 4px; }
#     ::-webkit-scrollbar-track { background: transparent; }
#     ::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.3); border-radius: 2px; }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# #  LOAD MODELS (cached)
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner=False)
# def load_chain():
#     """Load embeddings, vector DB, LLM, and build the QA chain."""

#     # 1. Embeddings
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL)

#     # 2. Vector store
#     if not os.path.exists(DB_DIR):
#         return None, "database_missing"

#     vectorstore = Chroma(
#         persist_directory=DB_DIR,
#         embedding_function=embeddings
#     )

#     # 3. LLM
#     llm = OllamaLLM(
#         model=LLM_MODEL,
#         temperature=TEMPERATURE,
#         num_predict=1024,
#     )

#     # 4. System prompt
#     system_prompt = """You are NUST Admissions Assistant — a helpful, accurate, and friendly guide for students applying to NUST (National University of Sciences and Technology), Pakistan.

# Your role:
# - Answer questions about NUST admissions, programs, eligibility, NET test, merit formula, fees, scholarships, hostels, and deadlines
# - Use ONLY the provided context to answer. Do not make up information.
# - If the context doesn't contain the answer, say: "I don't have that specific information. Please contact NUST Admissions Office directly at admissions@nust.edu.pk or call +92-51-90851001."
# - Be concise but complete. Use bullet points for lists.
# - Always be encouraging and student-friendly.
# - If asked about something unrelated to NUST, politely redirect.

# Context from NUST documents:
# {context}

# Chat History:
# {chat_history}"""

#     # 5. Memory (remembers last 5 exchanges)
#     memory = ConversationBufferWindowMemory(
#         k=5,
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer"
#     )

#     # 6. Conversational RAG chain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(
#             search_type="mmr",             # Max Marginal Relevance = diverse results
#             search_kwargs={"k": TOP_K_DOCS, "fetch_k": 15}
#         ),
#         memory=memory,
#         return_source_documents=True,
#         combine_docs_chain_kwargs={
#             "prompt": ChatPromptTemplate.from_messages([
#                 SystemMessagePromptTemplate.from_template(system_prompt),
#                 HumanMessagePromptTemplate.from_template("{question}")
#             ])
#         },
#         verbose=False,
#     )

#     return chain, "ok"


# # ─────────────────────────────────────────────
# #  SIDEBAR
# # ─────────────────────────────────────────────
# with st.sidebar:
#     st.markdown('<div class="sidebar-header">🎓 NUST Guide</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-sub">Admissions Assistant • Offline</div>', unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown("**Quick Questions**")

#     quick_questions = [
#         "What is the merit formula for BS programs?",
#         "When is the NET test scheduled?",
#         "What are the fee structures for UG?",
#         "What scholarships are available?",
#         "What are the eligibility criteria for MS?",
#         "How do I apply for PhD admission?",
#         "What programs does NUST offer?",
#         "Is hostel available for students?",
#         "What is the NET test syllabus?",
#         "How does the NUST scholarship work?",
#     ]

#     for q in quick_questions:
#         if st.button(q, key=f"quick_{q[:20]}"):
#             st.session_state.pending_question = q

#     st.markdown("---")

#     # Model info
#     st.markdown(f"""
#     <div style="font-size:12px; color:#8896a8; line-height:1.8;">
#         <div>🤖 <b style="color:#C9A84C">LLM:</b> {LLM_MODEL}</div>
#         <div>🔍 <b style="color:#C9A84C">Embed:</b> {EMBED_MODEL}</div>
#         <div>💾 <b style="color:#C9A84C">DB:</b> ChromaDB (local)</div>
#         <div>🔒 <b style="color:#C9A84C">Mode:</b> Fully offline</div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")

#     if st.button("🗑️ Clear Chat", key="clear"):
#         st.session_state.messages = []
#         st.session_state.pop("pending_question", None)
#         st.rerun()


# # ─────────────────────────────────────────────
# #  MAIN AREA
# # ─────────────────────────────────────────────
# st.markdown("""
# <div class="main-header">
#     <div class="main-title">NUST Admissions<br>Assistant</div>
#     <div class="main-subtitle">Your intelligent guide to admissions at NUST, Pakistan</div>
#     <div class="status-badge">
#         <div class="status-dot"></div>
#         Running Locally • No Internet Required
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────────
# #  LOAD CHAIN
# # ─────────────────────────────────────────────
# with st.spinner("Loading AI models..."):
#     chain, status = load_chain()

# if status == "database_missing":
#     st.error("""
#     ⚠️ **Vector database not found!**

#     You need to run the ingestion pipeline first:
#     ```
#     python ingest.py
#     ```
#     Make sure you've scraped data first with `python nust_scraper.py`
#     """)
#     st.stop()

# if chain is None:
#     st.error("❌ Failed to load the AI chain. Check that Ollama is running: `ollama serve`")
#     st.stop()

# # ─────────────────────────────────────────────
# #  CHAT STATE
# # ─────────────────────────────────────────────
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {
#             "role": "assistant",
#             "content": "👋 Hello! I'm your **NUST Admissions Assistant**.\n\nI can help you with:\n- 📋 Admission eligibility & criteria\n- 📝 NET test information\n- 💰 Fee structures & scholarships\n- 🏫 Programs offered\n- 📅 Important dates & deadlines\n- 🏠 Hostel & campus life\n\nWhat would you like to know about NUST admissions?",
#             "sources": []
#         }
#     ]

# # ─────────────────────────────────────────────
# #  RENDER MESSAGES
# # ─────────────────────────────────────────────
# chat_container = st.container()

# with chat_container:
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             with st.chat_message("user", avatar="👤"):
#                 st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
#         else:
#             with st.chat_message("assistant", avatar="🎓"):
#                 # Preserve newlines for markdown, but wrap in div for custom styling
#                 sources_html = ""
#                 if msg.get("sources"):
#                     unique_sources = list(set(msg["sources"]))[:4]
#                     sources_html = "<div style='margin-top:10px'>"
#                     for s in unique_sources:
#                         sources_html += f'<span class="source-tag">📄 {s}</span>'
#                     sources_html += "</div>"
                
#                 # Render content as markdown inside the bubble
#                 st.markdown(
#                     f'<div class="assistant-bubble">', 
#                     unsafe_allow_html=True
#                 )
#                 st.markdown(msg["content"])
#                 st.markdown(
#                     f'{sources_html}</div>',
#                     unsafe_allow_html=True
#                 )

# # ─────────────────────────────────────────────
# #  HANDLE INPUT
# # ─────────────────────────────────────────────
# def get_source_names(source_docs):
#     """Extract clean filenames from source documents."""
#     names = []
#     for doc in source_docs:
#         src = doc.metadata.get("source_file", "")
#         if src:
#             # Clean up filename for display
#             clean = src.replace("admissions__", "").replace("__.txt", "").replace("__", " › ").replace("_", " ").replace(".txt", "")
#             names.append(clean.title())
#     return list(set(names))


# def process_question(question):
#     """Run the question through the RAG chain and return answer + sources."""
#     try:
#         result = chain({"question": question})
#         answer = result.get("answer", "I couldn't generate a response. Please try again.")
#         sources = get_source_names(result.get("source_documents", []))
#         return answer, sources
#     except Exception as e:
#         return f"⚠️ Error: {str(e)}\n\nMake sure Ollama is running with `ollama serve`", []


# # Check for quick question button press
# pending = st.session_state.pop("pending_question", None)

# # Chat input
# user_input = st.chat_input("Ask about NUST admissions, programs, fees, scholarships...") or pending

# if user_input:
#     # Add user message to history
#     st.session_state.messages.append({
#         "role": "user",
#         "content": user_input,
#         "sources": []
#     })

#     # Show new messages
#     with chat_container:
#         # 1. User Message
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

#         # 2. Assistant Message
#         with st.chat_message("assistant", avatar="🎓"):
#             with st.spinner("Searching NUST knowledge base..."):
#                 answer, sources = process_question(user_input)

#             # Preserve newlines and render as markdown
#             sources_html = ""
#             if sources:
#                 sources_html = "<div style='margin-top:10px'>"
#                 for s in sources[:4]:
#                     sources_html += f'<span class="source-tag">📄 {s}</span>'
#                 sources_html += "</div>"
            
#             st.markdown(f'<div class="assistant-bubble">', unsafe_allow_html=True)
#             st.markdown(answer)
#             st.markdown(f'{sources_html}</div>', unsafe_allow_html=True)

#     # Save assistant message to history
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer,
#         "sources": sources
#     })

"""
NUST Admissions Chatbot
========================
Compatible with: LangChain 0.3+, Python 3.13, Ollama (Mistral)
Run: streamlit run chatbot.py
"""

import streamlit as st
import os

# ── LangChain 0.3+ compatible imports ─────────────────────
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DB_DIR      = "./nust_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "tinyllama"        # Change to llama3.1:8b if needed
TEMPERATURE = 0.1
TOP_K_DOCS  = 5

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NUST Admissions Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --nust-green: #006633;
        --nust-gold:  #C9A84C;
        --nust-dark:  #0a1628;
        --nust-mid:   #112240;
        --text-main:  #e8edf5;
        --text-muted: #8896a8;
    }

    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%);
        border-right: 1px solid rgba(201,168,76,0.2);
    }

    .user-bubble {
        background: linear-gradient(135deg, #006633, #004d26);
        border: 1px solid rgba(201,168,76,0.3);
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        color: #e8edf5;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        line-height: 1.6;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 20px rgba(0,102,51,0.3);
    }

    .assistant-bubble {
        background: linear-gradient(135deg, #112240, #1a2d50);
        border: 1px solid rgba(79,195,247,0.2);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        color: #e8edf5;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        line-height: 1.6;
        max-width: 90%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .assistant-bubble strong { color: #C9A84C; }
    .assistant-bubble ul { padding-left: 20px; }
    .assistant-bubble li { margin: 6px 0; }

    .source-tag {
        display: inline-block;
        background: rgba(201,168,76,0.1);
        border: 1px solid rgba(201,168,76,0.3);
        color: #C9A84C;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 20px;
        margin: 4px 3px 0 0;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 38px;
        background: linear-gradient(135deg, #ffffff 0%, #C9A84C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        text-align: center;
        margin-bottom: 6px;
    }
    .main-subtitle {
        color: #8896a8;
        font-size: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-badge {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 6px;
        color: #4caf82;
        font-size: 12px;
        margin-bottom: 24px;
    }
    .status-dot {
        width: 7px; height: 7px;
        background: #4caf82;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%,100% { opacity:1; } 50% { opacity:0.3; }
    }

    .stButton > button {
        background: rgba(17,34,64,0.8) !important;
        border: 1px solid rgba(79,195,247,0.2) !important;
        color: #8896a8 !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 6px 14px !important;
        width: 100% !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        border-color: rgba(201,168,76,0.5) !important;
        color: #C9A84C !important;
    }

    hr { border-color: rgba(201,168,76,0.15) !important; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS (cached — loads only once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_retriever():
    if not os.path.exists(DB_DIR):
        return None, "db_missing"
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K_DOCS, "fetch_k": 15}
        )
        return retriever, "ok"
    except Exception as e:
        return None, str(e)


# @st.cache_resource(show_spinner=False)
# def load_llm():
#     try:
#         llm = OllamaLLM(model=LLM_MODEL, temperature=TEMPERATURE)
#         return llm, "ok"
#     except Exception as e:
#         return None, str(e)

@st.cache_resource(show_spinner=False)
def load_llm():
    try:
        llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            num_ctx=512,          # very small context = less RAM
            num_predict=256,      # shorter answers = less RAM
            num_thread=4,         # limit CPU threads
        )
        return llm, "ok"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
#  RAG CHAIN
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are the NUST Admissions Assistant — a helpful, accurate, and friendly guide for students applying to NUST (National University of Sciences and Technology), Pakistan.

Rules:
- Answer ONLY using the provided context below.
- If the answer is not in the context, say: "I don't have that specific information. Please contact NUST Admissions at admissions@nust.edu.pk or call 111-11-NUST."
- Be concise, clear, and student-friendly.
- Use bullet points for lists.
- Never make up information.

Context from NUST documents:
{context}

Previous conversation:
{chat_history}

Student question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_source_names(docs):
    names = []
    for doc in docs:
        src = doc.metadata.get("source_file", "")
        if src:
            clean = (src.replace("admissions__", "")
                       .replace("__.txt", "")
                       .replace("__", " › ")
                       .replace("_", " ")
                       .replace(".txt", ""))
            names.append(clean.title())
    return list(set(names))


def format_chat_history(messages):
    history = ""
    for msg in messages[-6:]:
        role = "Student" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history.strip()


def ask_question(question, retriever, llm, chat_history_text):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history_text,
        "question": question,
    })
    return answer, docs


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:22px;color:#C9A84C;">
        🎓 NUST Guide
    </div>
    <div style="font-size:13px;color:#8896a8;margin-bottom:16px;">
        Admissions Assistant • Offline
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Quick Questions**")

    quick_questions = [
        "What is the merit formula for BS?",
        "When is the NET test?",
        "What are the UG fee structures?",
        "What scholarships are available?",
        "Eligibility criteria for MS admission?",
        "How to apply for PhD?",
        "What programs does NUST offer?",
        "Is hostel available for students?",
        "What is the NET test syllabus?",
        "How does NUST scholarship work?",
    ]

    for q in quick_questions:
        if st.button(q, key=f"q_{q[:15]}"):
            st.session_state.pending_question = q

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:12px;color:#8896a8;line-height:2;">
        🤖 <b style="color:#C9A84C">Model:</b> {LLM_MODEL}<br>
        🔍 <b style="color:#C9A84C">Embed:</b> {EMBED_MODEL}<br>
        💾 <b style="color:#C9A84C">DB:</b> ChromaDB (local)<br>
        🔒 <b style="color:#C9A84C">Mode:</b> Fully offline
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">NUST Admissions Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Your intelligent guide to admissions at NUST, Pakistan</div>', unsafe_allow_html=True)
st.markdown('<div class="status-badge"><span class="status-dot"></span> Running Locally</div>', unsafe_allow_html=True)

# Load models
with st.spinner("Loading AI models... (first load may take ~30 seconds)"):
    retriever, r_status = load_retriever()
    llm, l_status = load_llm()

if r_status == "db_missing":
    st.error("⚠️ **Vector database not found!**")
    st.info("Run this command first:\n```\npython ingest.py\n```")
    st.stop()

if retriever is None:
    st.error(f"❌ Failed to load database: {r_status}")
    st.stop()

if llm is None:
    st.error(f"❌ Failed to load Ollama model.\n\nMake sure Ollama is running:\n```\nollama serve\n```")
    st.stop()

# ── Chat state ────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your **NUST Admissions Assistant**.\n\nI can help you with:\n- 📋 Admission eligibility & criteria\n- 📝 NET test information\n- 💰 Fee structures & scholarships\n- 🏫 Programs offered at NUST\n- 📅 Important dates & deadlines\n- 🏠 Hostel & campus life\n\nWhat would you like to know?",
            "sources": []
        }
    ]

# ── Render existing messages ──────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🎓"):
            content = msg["content"].replace("\n", "<br>")
            sources_html = ""
            if msg.get("sources"):
                sources_html = "<div style='margin-top:10px'>"
                for s in msg["sources"][:4]:
                    sources_html += f'<span class="source-tag">📄 {s}</span>'
                sources_html += "</div>"
            st.markdown(
                f'<div class="assistant-bubble">{content}{sources_html}</div>',
                unsafe_allow_html=True
            )

# ── Chat input ────────────────────────────────
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask about NUST admissions, programs, fees, scholarships...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})

    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching NUST knowledge base..."):
            try:
                history_text = format_chat_history(st.session_state.messages[:-1])
                answer, source_docs = ask_question(user_input, retriever, llm, history_text)
                sources = get_source_names(source_docs)
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}\n\nMake sure `ollama serve` is running in a separate terminal."
                sources = []

        content = answer.replace("\n", "<br>")
        sources_html = ""
        if sources:
            sources_html = "<div style='margin-top:10px'>"
            for s in sources[:4]:
                sources_html += f'<span class="source-tag">📄 {s}</span>'
            sources_html += "</div>"
        st.markdown(
            f'<div class="assistant-bubble">{content}{sources_html}</div>',
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})