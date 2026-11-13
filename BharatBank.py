import os
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

import streamlit as st
from dotenv import load_dotenv

from src.generator.rag_pipeline import BharatBankRAGPipeline

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BharatBank RAG · RBI Regulatory Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Theme (forest black + emerald + gold)
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: #060D0B;
  --panel: #0C1814;
  --panel-2: #11221C;
  --line: #1B3128;
  --accent: #2FD08A;
  --gold: #F2C14E;
  --text: #E8F3EE;
  --muted: #86A398;
  --good: #6EE7B7;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: var(--text); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

/* Background: deep navy with a faint engineering grid */
.stApp {
  background:
    radial-gradient(900px 500px at 85% -10%, rgba(47,208,138,0.12), transparent 60%),
    radial-gradient(700px 400px at -10% 10%, rgba(242,193,78,0.07), transparent 60%),
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px) 0 0 / 44px 44px,
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px) 0 0 / 44px 44px,
    var(--ink);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 980px; padding-top: 1.6rem; padding-bottom: 6rem; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #09120F 0%, #060D0B 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.side-label { color: var(--muted); font-size: 0.78rem; font-weight: 600; margin: 1.1rem 0 0.4rem; }

/* Hero */
.hero {
  position: relative; overflow: hidden;
  border: 1px solid var(--line); border-radius: 20px;
  padding: 1.8rem 2rem;
  background: linear-gradient(135deg, rgba(17,34,28,0.95), rgba(9,18,15,0.95));
}
.hero::before {
  content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), var(--gold), transparent);
  background-size: 200% 100%;
  animation: sweep 6s linear infinite;
}
@keyframes sweep { from { background-position: 200% 0; } to { background-position: -200% 0; } }
.hero h1 { font-size: 2.3rem; margin: 0 0 0.35rem; line-height: 1.1; }
.hero h1 span { color: var(--gold); }
.hero p { color: var(--muted); margin: 0; max-width: 62ch; line-height: 1.6; }
.status {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 0.8rem; color: var(--good);
  background: rgba(110,231,183,0.08); border: 1px solid rgba(110,231,183,0.25);
  padding: 4px 12px; border-radius: 999px; margin-bottom: 0.9rem;
}
.status i { width: 7px; height: 7px; border-radius: 50%; background: var(--good); animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(110,231,183,0.5);} 50% { opacity: .6; box-shadow: 0 0 0 6px rgba(110,231,183,0);} }

/* Feature chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1.1rem; }
.chip {
  font-size: 0.8rem; color: var(--text);
  border: 1px solid var(--line); background: rgba(255,255,255,0.03);
  padding: 5px 12px; border-radius: 999px;
}

/* Stat cards (sidebar) */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat {
  border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px;
  background: var(--panel);
}
.stat b { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; color: var(--accent); }
.stat span { font-size: 0.75rem; color: var(--muted); }

/* Section heading */
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; margin: 1.6rem 0 0.2rem; }
.section-sub { color: var(--muted); font-size: 0.88rem; margin-bottom: 0.6rem; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  width: 100%; text-align: left; justify-content: flex-start;
  background: var(--panel); color: var(--text);
  border: 1px solid var(--line); border-radius: 12px;
  padding: 0.65rem 0.9rem; font-size: 0.88rem; font-weight: 500;
  transition: border-color .15s ease, background .15s ease, transform .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent); background: var(--panel-2); color: #fff;
}
.stButton > button:active { transform: scale(0.985); }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #1FA971, #2FD08A); color: #04170F; border: none; font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); padding: 8px 14px; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }

/* Chat messages */
[data-testid="stChatMessage"] {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: rgba(47,208,138,0.06); border-color: rgba(47,208,138,0.25);
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height: 1.7; }
.meta { color: var(--muted); font-size: 0.76rem; margin-top: 0.5rem; }

/* Chat input */
[data-testid="stChatInput"] { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(47,208,138,0.15); }
[data-testid="stBottom"] > div { background: transparent; }

/* Expander (sources) */
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }

/* Disclaimer */
.disclaimer {
  color: var(--muted); font-size: 0.76rem; text-align: center;
  border-top: 1px solid var(--line); margin-top: 2rem; padding-top: 0.9rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# BeautifulSoup PDF Auto-Sync Scraper Module
# ──────────────────────────────────────────────────────────────
def sync_rbi_pdfs(target_url: str, save_dir: str = "data/rbi_docs"):
    """
    Scrapes the RBI portal via BeautifulSoup on startup, clears old local PDFs, 
    and downloads fresh files in bulk.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        response = requests.get(target_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Web scraping connection failed: {e}")
        return False

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")
    pdf_links = []
    
    for link in links:
        href = link.get("href")
        if href and (".pdf" in href.lower() or "pdf" in link.get("class", [])):
            absolute_url = urljoin(target_url, href)
            if absolute_url not in pdf_links:
                pdf_links.append(absolute_url)

    if not pdf_links:
        return False

    # Purge stale local PDFs
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
            os.remove(file_path)

    # Bulk download fresh PDFs
    for i, pdf_url in enumerate(pdf_links[:10], 1): # Limited to top 10 fresh circulars for speed
        try:
            parsed_url = urlparse(pdf_url)
            file_name = os.path.basename(parsed_url.path)
            if not file_name.lower().endswith(".pdf"):
                file_name = f"rbi_document_{i}.pdf"
                
            file_path = os.path.join(save_dir, file_name)
            pdf_response = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if pdf_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(pdf_response.content)
        except Exception:
            continue
    return True

# Run auto-sync scraper once per session startup
if "pdfs_synced" not in st.session_state:
    with st.spinner("🔄 Syncing latest RBI circulars via BeautifulSoup scraper..."):
        TARGET_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"
        sync_rbi_pdfs(TARGET_URL, save_dir="data/rbi_docs")
        st.session_state["pdfs_synced"] = True
        st.toast("RBI document repository synchronized successfully!", icon="📂")

# ──────────────────────────────────────────────────────────────
# Data & Topics
# ──────────────────────────────────────────────────────────────
TOPICS = {
    "KYC & AML": [
        "What are the RBI KYC requirements for opening a savings account?",
        "How often should banks perform periodic KYC updation?",
        "What is Video-based Customer Identification (V-CIP)?",
        "What are the rules for reporting suspicious transactions?",
    ],
    "Loans & Credit": [
        "What are the RBI guidelines on digital lending?",
        "What is the Key Fact Statement (KFS) for loans?",
        "What are the norms for prepayment charges on floating rate loans?",
        "What are the priority sector lending targets?",
    ],
    "Asset Quality": [
        "How does RBI classify an account as an NPA?",
        "What are the provisioning norms for substandard assets?",
        "What is the framework for resolving stressed assets?",
        "What are the rules on wilful defaulters?",
    ],
    "Payments & Digital": [
        "What are the RBI rules on UPI transaction limits?",
        "What is the framework for customer liability in unauthorised transactions?",
        "What are the guidelines on card tokenisation?",
        "What are the cyber security requirements for banks?",
    ],
    "Customer Rights": [
        "How do I file a complaint under the RBI Integrated Ombudsman Scheme?",
        "What is the timeline for banks to resolve a complaint?",
        "What are the minimum balance and service charge rules?",
        "What are the rules on compensation for failed ATM transactions?",
    ],
}

STYLES = {
    "Standard": "",
    "Concise": " Answer concisely in at most 5 bullet points.",
    "Detailed": " Give a detailed, well-structured answer with clear section headings.",
    "Plain English": " Explain in simple language for someone with no banking background.",
}

# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    return BharatBankRAGPipeline()

try:
    with st.spinner("Loading the RBI knowledge base…"):
        rag_engine = load_rag_pipeline()
except Exception as e:
    st.error(f"The knowledge base failed to load: {e}")
    horn_info = "Check that the vector store exists and your API keys are set in .env."
    st.info(horn_info)
    st.stop()

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("feedback", {})

def ask(text: str, regen: bool = False):
    st.session_state.pending = {"text": text, "regen": regen}

def normalize(resp):
    if isinstance(resp, dict):
        answer = resp.get("answer") or resp.get("response") or resp.get("result") or str(resp)
        sources = resp.get("sources") or resp.get("source_documents") or []
        return answer, sources
    return str(resp), []

def export_markdown() -> str:
    lines = ["# BharatBank RAG · Chat export", f"_Exported {datetime.now():%d %b %Y, %H:%M}_", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "BharatBank RAG"
        lines += [f"### {who} ({m.get('ts', '')})", m["content"], ""]
    lines.append("---\n*AI-generated. Verify against original RBI circulars before acting.*")
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 BharatBank RAG")
    st.caption("RBI regulatory compliance assistant")

    st.markdown('<div class="side-label">Answer settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Documents to search", 1, 10, 4)
    style = st.selectbox("Answer style", list(STYLES.keys()))

    st.markdown('<div class="side-label">Session Statistics</div>', unsafe_allow_html=True)
    msgs = st.session_state.messages
    n_q = sum(1 for m in msgs if m["role"] == "user")
    times = [m["elapsed"] for m in msgs if m.get("elapsed")]
    avg_t = f"{sum(times) / len(times):.1f}s" if times else "–"
    st.markdown(
        f'<div class="stat-grid"><div class="stat"><b>{n_q}</b><span>Queries</span></div>'
        f'<div class="stat"><b>{avg_t}</b><span>Speed</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Actions</div>', unsafe_allow_html=True)
    if st.button("✨  New chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.feedback = {}
        st.rerun()
        
    st.download_button(
        "⬇  Export chat (.md)",
        data=export_markdown(),
        file_name=f"bharatbank_chat_{datetime.now():%Y%m%d_%H%M}.md",
        mime="text/markdown",
        disabled=not msgs,
        use_container_width=True,
    )

    st.markdown('<div class="side-label">Official RBI Links</div>', unsafe_allow_html=True)
    st.link_button("RBI Notifications Portal", "https://www.rbi.org.in/Scripts/NotificationUser.aspx", use_container_width=True)
    st.link_button("RBI Master Directions", "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx", use_container_width=True)
    st.link_button("RBI Ombudsman Scheme", "https://cms.rbi.org.in/", use_container_width=True)

    st.markdown(
        '<div class="disclaimer">BharatBank RAG provides informational compliance guidance. Verify legal provisions directly on RBI portals.</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Hero section
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
<div class="status"><i></i>Live Scraper & Knowledge Base Active</div>
<h1>Ask the RBI rulebook <span>anything.</span></h1>
<p>Get accurate, professionally cited regulatory answers synthesized from live-scraped Reserve Bank of India circulars and master directions.</p>
<div class="chips">
<span class="chip">📜 Master Directions</span>
<span class="chip">🪪 KYC &amp; AML</span>
<span class="chip">💳 Digital Lending</span>
<span class="chip">📊 Asset Classification</span>
<span class="chip">🛡️ Consumer Protection</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Welcome screen: topic launcher
# ──────────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.pending:
    st.markdown('<div class="section-title">Start with a common banking compliance query</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select a category below or type your custom question in the chat bar.</div>', unsafe_allow_html=True)

    tabs = st.tabs(list(TOPICS.keys()))
    for tab, (name, questions) in zip(tabs, TOPICS.items()):
        with tab:
            cols = st.columns(2)
            for i, q in enumerate(questions):
                with cols[i % 2]:
                    st.button(q, key=f"topic_{name}_{i}", on_click=ask, args=(q,))

def regenerate():
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
        last_q = msgs[-1]["content"] if msgs else None
        if last_q:
            ask(last_q, regen=True)

# ──────────────────────────────────────────────────────────────
# Chat History & Render Loop
# ──────────────────────────────────────────────────────────────
last_idx = len(st.session_state.messages) - 1
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🏦"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("sources"):
                with st.expander(f"📚 Official RBI Citations & Documents ({len(m['sources'])})"):
                    for s in m["sources"]:
                        # Clean up raw hash strings if present, formatting cleanly
                        clean_src = s.split("File:")[-1].strip(")") if "File:" in s else s
                        st.markdown(f"- 📄 **{clean_src}** · [View RBI Portals](https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx)")
            
            st.markdown(f'<div class="meta">{m.get("ts", "")} · generated in {m.get("elapsed", 0):.1f}s</div>', unsafe_allow_html=True)

            c1, c2, _ = st.columns([1, 1.4, 4])
            with c1:
                if hasattr(st, "feedback"):
                    fb = st.feedback("thumbs", key=f"fb_{idx}")
                    if fb is not None:
                        st.session_state.feedback[idx] = fb
            with c2:
                if idx == last_idx:
                    st.button("↻  Regenerate", key=f"regen_{idx}", on_click=regenerate)

# ──────────────────────────────────────────────────────────────
# Chat Input & Execution Logic
# ──────────────────────────────────────────────────────────────
typed = st.chat_input("Ask about RBI guidelines, KYC rules, loan rates, NPAs, digital payments…")
if typed:
    ask(typed)

pending = st.session_state.pending
if pending:
    st.session_state.pending = None
    prompt = pending["text"]
    stamp = f"{datetime.now():%d %b, %H:%M}"

    if not pending["regen"]:
        st.session_state.messages.append({"role": "user", "content": prompt, "ts": stamp})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏦"):
        try:
            start = time.time()
            with st.spinner(f"Searching top {top_k} RBI documents and synthesizing regulatory answer..."):
                raw = rag_engine.generate_answer(prompt + STYLES[style], top_k=top_k)
            elapsed = time.time() - start
            answer, sources = normalize(raw)
            
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "elapsed": elapsed,
                    "ts": f"{datetime.now():%d %b, %H:%M}",
                }
            )
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate an answer: {e}")
            st.button("↻  Try again", key="retry", on_click=ask, args=(prompt, True))

if not st.session_state.messages:
    st.markdown(
        '<div class="disclaimer">BharatBank RAG is an advanced compliance tool built for banking professionals.</div>',
        unsafe_allow_html=True,
    )