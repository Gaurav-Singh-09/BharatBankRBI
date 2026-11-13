import os
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.generator.rag_pipeline import BharatBankRAGPipeline

load_dotenv()

DOCS_DIR = "data/rbi_docs"
SYNC_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"

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
# Theme: indigo + saffron, with dark and light palettes
# ──────────────────────────────────────────────────────────────
THEMES = {
    "dark": dict(bg="#0A0E1F", bg2="#0F1530", panel="#131A38", panel2="#1A2247", line="#262F5C",
                 text="#EEF0FF", muted="#8E98C9", ac="#8B9BFF", ac2="#FFB454", ok="#5EEAD4",
                 glow="rgba(139,155,255,0.16)", on="#0A0E1F", btn_on="#1b1100"),
    "light": dict(bg="#F6F5FF", bg2="#ECEBFF", panel="#FFFFFF", panel2="#F0F0FF", line="#DAD9F5",
                  text="#171A3A", muted="#5F668F", ac="#4C5CE0", ac2="#C46A0A", ok="#0F8F7E",
                  glow="rgba(76,92,224,0.12)", on="#FFFFFF", btn_on="#FFFFFF"),
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; color: var(--text); }
h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; color: var(--text); }
.stApp { background: radial-gradient(800px 420px at 90% -8%, var(--glow), transparent 60%), var(--bg); }
.stApp p, .stApp li, .stApp label, .stApp span { color: inherit; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 980px; padding-top: 1.2rem; padding-bottom: 7rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { display: flex; gap: 10px; align-items: center; margin-bottom: .4rem; }
.logo { width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center; font-size: 1.2rem;
        background: linear-gradient(140deg, var(--ac), var(--ac2)); color: var(--on); }
.brand b { font-family: 'Fraunces', serif; font-size: 1.15rem; display: block; line-height: 1.1; }
.brand small { color: var(--muted); font-size: .74rem; }
.side-label { color: var(--muted); font-size: .78rem; font-weight: 700; margin: 1.1rem 0 .4rem; }

/* Top bar */
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: .4rem; gap: .8rem; flex-wrap: wrap; }
.crumb { color: var(--muted); font-size: .85rem; }
.pill { display: inline-flex; align-items: center; gap: 8px; font-size: .78rem; font-weight: 600; padding: 4px 12px; border-radius: 999px;
        color: var(--ok); border: 1px solid color-mix(in srgb, var(--ok) 35%, transparent); background: color-mix(in srgb, var(--ok) 9%, transparent); }
.pill i { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); }
.pill.warn { color: var(--ac2); border-color: color-mix(in srgb, var(--ac2) 35%, transparent); background: color-mix(in srgb, var(--ac2) 9%, transparent); }
.pill.warn i { background: var(--ac2); }

/* Hero */
.hero { padding: 1.6rem 0 .6rem; }
.hero h1 { font-size: clamp(2rem, 4.5vw, 3.1rem); line-height: 1.08; margin: 0 0 .7rem; max-width: 18ch; }
.hero p { color: var(--muted); max-width: 58ch; margin: 0 0 1rem; font-size: 1.02rem; line-height: 1.6; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-size: .8rem; border: 1px solid var(--line); background: var(--panel); padding: 5px 12px; border-radius: 999px; }

/* Stat strip */
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1.3rem 0; }
.stat-card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem; }
.stat-card b { font-family: 'Fraunces', serif; font-size: 1.9rem; display: block; line-height: 1.1; }
.stat-card span { color: var(--muted); font-size: .82rem; }

/* Sidebar stat cards */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat { border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; background: var(--panel); }
.stat b { display: block; font-family: 'Fraunces', serif; font-size: 1.35rem; color: var(--ac); }
.stat span { font-size: .75rem; color: var(--muted); }

/* Cards, section headings */
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.2rem; }
.card h3 { font-size: 1.1rem; margin: 0 0 .6rem; }
.doc { display: flex; gap: 10px; padding: .55rem 0; border-bottom: 1px solid var(--line); font-size: .84rem; word-break: break-all; }
.doc:last-child { border: 0; }
.tag { flex: none; align-self: flex-start; font-size: .66rem; font-weight: 700; border-radius: 6px; padding: 2px 7px; background: var(--glow); color: var(--ac); }
.section-title { font-family: 'Fraunces', serif; font-size: 1.15rem; margin: 1.2rem 0 .2rem; }
.section-sub { color: var(--muted); font-size: .88rem; margin-bottom: .6rem; }

/* Buttons */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
  width: 100%; text-align: left; justify-content: flex-start; background: var(--panel); color: var(--text);
  border: 1px solid var(--line); border-radius: 12px; padding: .6rem .9rem; font-size: .88rem; font-weight: 600;
  transition: border-color .15s ease, background .15s ease, transform .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover { border-color: var(--ac); background: var(--panel2); color: var(--text); }
.stButton > button:active { transform: scale(.985); }
.stButton > button:focus-visible { outline: 2px solid var(--ac); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--ac); color: var(--on); border: none; font-weight: 700; }
.stButton > button p, .stLinkButton > a p { color: inherit; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); padding: 8px 14px; border-radius: 8px 8px 0 0; font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--ac) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--ac); }

/* Chat */
[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.2rem; margin-bottom: .8rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--ac); border-color: var(--ac); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: var(--on) !important; font-weight: 600; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height: 1.7; }
.meta { color: var(--muted); font-size: .76rem; margin-top: .5rem; }
.src { display: flex; flex-wrap: wrap; gap: 8px; margin: .6rem 0; }
.src span { border: 1px solid var(--line); background: var(--bg2); border-radius: 10px; padding: .3rem .7rem; font-size: .78rem; }
.src b { color: var(--ac); margin-right: 4px; }

/* Chat input */
[data-testid="stChatInput"] { border: 1px solid var(--line); border-radius: 18px; background: var(--panel); }
[data-testid="stChatInput"]:focus-within { border-color: var(--ac); box-shadow: 0 0 0 4px var(--glow); }
[data-testid="stChatInput"] textarea { color: var(--text); }
[data-testid="stBottom"] > div { background: transparent; }

/* Expander, inputs */
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
[data-baseweb="select"] > div { background: var(--panel); border-color: var(--line); }

.disclaimer { color: var(--muted); font-size: .76rem; text-align: center; border-top: 1px solid var(--line); margin-top: 1.6rem; padding-top: .9rem; }

@media (max-width: 760px) { .stats { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: no-preference) { .hero h1 { animation: rise .7s ease both; } @keyframes rise { from { opacity: 0; transform: translateY(10px); } } }
"""

def inject_theme(mode: str):
    t = THEMES[mode]
    var_map = {"bg": "bg", "bg2": "bg2", "panel": "panel", "panel2": "panel2", "line": "line", "text": "text",
               "muted": "muted", "ac": "ac", "ac2": "ac2", "ok": "ok", "glow": "glow", "on": "on"}
    root = ":root{" + "".join(f"--{k}:{t[v]};" for k, v in var_map.items()) + "}"
    st.markdown(f"<style>{root}{BASE_CSS}</style>", unsafe_allow_html=True)

st.session_state.setdefault("theme", "dark")
inject_theme(st.session_state.theme)

# ──────────────────────────────────────────────────────────────
# RBI PDF sync
# ──────────────────────────────────────────────────────────────
def list_local_pdfs(save_dir: str = DOCS_DIR):
    if not os.path.isdir(save_dir):
        return []
    return sorted(f for f in os.listdir(save_dir) if f.lower().endswith(".pdf"))

def sync_rbi_pdfs(target_url: str, save_dir: str = DOCS_DIR, limit: int = 10) -> int:
    """Download up to `limit` PDFs linked from the RBI page. Returns how many were saved.
    Old PDFs are only replaced if at least one new file downloaded, so a failed sync never empties the folder."""
    os.makedirs(save_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(target_url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Web scraping connection failed: {e}")
        return 0

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and ".pdf" in href.lower():
            absolute = urljoin(target_url, href)
            if absolute not in pdf_links:
                pdf_links.append(absolute)
    if not pdf_links:
        return 0

    fresh = []
    for i, pdf_url in enumerate(pdf_links[:limit], 1):
        try:
            name = os.path.basename(urlparse(pdf_url).path)
            if not name.lower().endswith(".pdf"):
                name = f"rbi_document_{i}.pdf"
            r = requests.get(pdf_url, headers=headers, timeout=20)
            if r.status_code == 200 and r.content:
                fresh.append((name, r.content))
        except Exception:
            continue
    if not fresh:
        return 0

    for f in list_local_pdfs(save_dir):
        os.remove(os.path.join(save_dir, f))
    for name, content in fresh:
        with open(os.path.join(save_dir, name), "wb") as fh:
            fh.write(content)
    return len(fresh)

if "pdfs_synced" not in st.session_state:
    with st.spinner("🔄 Syncing latest RBI circulars…"):
        st.session_state["sync_count"] = sync_rbi_pdfs(SYNC_URL)
        st.session_state["synced_at"] = datetime.now()
        st.session_state["pdfs_synced"] = True
    if st.session_state["sync_count"]:
        st.toast(f"Synced {st.session_state['sync_count']} RBI documents", icon="📂")
    else:
        st.toast("Couldn't reach the RBI site. Using the documents already on disk.", icon="⚠️")

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

FOLLOW_UPS = ["Explain this in simple terms", "Give me a practical example", "What are the exceptions?"]

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
    st.info("Check that the vector store exists and your API keys are set in .env.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Session state & helpers
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

def clean_source(s) -> str:
    s = str(s)
    return s.split("File:")[-1].strip(") ").strip() if "File:" in s else s

def export_markdown() -> str:
    lines = ["# BharatBank RAG · Chat export", f"_Exported {datetime.now():%d %b %Y, %H:%M}_", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "BharatBank RAG"
        lines += [f"### {who} ({m.get('ts', '')})", m["content"], ""]
    lines.append("---\n*AI-generated. Verify against original RBI circulars before acting.*")
    return "\n".join(lines)

def regenerate():
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
        if msgs:
            ask(msgs[-1]["content"], regen=True)

def reset_chat():
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.feedback = {}

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">🏦</div><div><b>BharatBank RAG</b><small>RBI compliance assistant</small></div></div>',
        unsafe_allow_html=True,
    )
    st.button("＋  New chat", type="primary", use_container_width=True, on_click=reset_chat)

    msgs = st.session_state.messages
    user_qs = [m["content"] for m in msgs if m["role"] == "user"]
    if user_qs:
        st.markdown('<div class="side-label">This session</div>', unsafe_allow_html=True)
        for i, q in enumerate(reversed(user_qs[-5:])):
            st.button(("💬  " + q)[:44] + ("…" if len(q) > 40 else ""), key=f"recent_{i}", on_click=ask, args=(q,), use_container_width=True)

    st.markdown('<div class="side-label">Answer settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Documents to search", 1, 10, 4)
    style = st.selectbox("Answer style", list(STYLES.keys()))
    light = st.toggle("Light theme", value=st.session_state.theme == "light")
    new_theme = "light" if light else "dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    n_q = len(user_qs)
    times = [m["elapsed"] for m in msgs if m.get("elapsed")]
    avg_t = f"{sum(times) / len(times):.1f}s" if times else "–"
    st.markdown('<div class="side-label">Session statistics</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="stat-grid"><div class="stat"><b>{n_q}</b><span>Queries</span></div>'
        f'<div class="stat"><b>{avg_t}</b><span>Avg. speed</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Actions</div>', unsafe_allow_html=True)
    st.download_button(
        "⬇  Export chat (.md)",
        data=export_markdown(),
        file_name=f"bharatbank_chat_{datetime.now():%Y%m%d_%H%M}.md",
        mime="text/markdown",
        disabled=not msgs,
        use_container_width=True,
    )

    st.markdown('<div class="side-label">Official RBI links</div>', unsafe_allow_html=True)
    st.link_button("RBI Notifications Portal", "https://www.rbi.org.in/Scripts/NotificationUser.aspx", use_container_width=True)
    st.link_button("RBI Master Directions", SYNC_URL, use_container_width=True)
    st.link_button("RBI Ombudsman Scheme", "https://cms.rbi.org.in/", use_container_width=True)

    st.markdown(
        '<div class="disclaimer">Informational compliance guidance only. Verify legal provisions on the RBI portals.</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Top bar + hero
# ──────────────────────────────────────────────────────────────
local_pdfs = list_local_pdfs()
synced_at = st.session_state.get("synced_at")
if st.session_state.get("sync_count"):
    status = f'<span class="pill"><i></i>Synced {st.session_state["sync_count"]} documents at {synced_at:%H:%M}</span>'
elif local_pdfs:
    status = f'<span class="pill warn"><i></i>Offline sync · using {len(local_pdfs)} saved documents</span>'
else:
    status = '<span class="pill warn"><i></i>No documents synced yet</span>'

in_chat = bool(st.session_state.messages or st.session_state.pending)
st.markdown(
    f'<div class="topbar"><span class="crumb">Home{" / Current chat" if in_chat else ""}</span>{status}</div>',
    unsafe_allow_html=True,
)

if not in_chat:
    st.markdown(
        """
<div class="hero">
<h1>Every RBI circular, answered in plain words.</h1>
<p>Ask a compliance question and get an answer with the exact master direction behind it, drawn from circulars synced from the RBI portal.</p>
<div class="chips">
<span class="chip">📜 Master Directions</span><span class="chip">🪪 KYC &amp; AML</span><span class="chip">💳 Digital Lending</span>
<span class="chip">📊 Asset Classification</span><span class="chip">🛡️ Consumer Protection</span>
</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stats"><div class="stat-card"><b>{len(local_pdfs)}</b><span>Documents on disk</span></div>'
        f'<div class="stat-card"><b>{len(TOPICS)}</b><span>Topic areas to explore</span></div>'
        f'<div class="stat-card"><b>{top_k}</b><span>Documents searched per answer</span></div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown('<div class="section-title">Popular questions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Pick a category, or type your own question below.</div>', unsafe_allow_html=True)
        tabs = st.tabs(list(TOPICS.keys()))
        for tab, (name, questions) in zip(tabs, TOPICS.items()):
            with tab:
                for i, q in enumerate(questions):
                    st.button("↗  " + q, key=f"topic_{name}_{i}", on_click=ask, args=(q,))
    with right:
        st.markdown('<div class="section-title">Latest synced documents</div>', unsafe_allow_html=True)
        if local_pdfs:
            rows = "".join(f'<div class="doc"><span class="tag">PDF</span><span>{f}</span></div>' for f in local_pdfs[:6])
            more = f'<div class="section-sub" style="margin:.6rem 0 0">+ {len(local_pdfs) - 6} more</div>' if len(local_pdfs) > 6 else ""
            st.markdown(f'<div class="card">{rows}{more}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="section-sub">No PDFs found in data/rbi_docs yet.</div></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────
last_idx = len(st.session_state.messages) - 1
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🏦"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("sources"):
                chips = "".join(f'<span><b>[{n}]</b>{clean_source(s)}</span>' for n, s in enumerate(m["sources"], 1))
                st.markdown(f'<div class="src">{chips}</div>', unsafe_allow_html=True)
                with st.expander(f"📚 Official RBI citations ({len(m['sources'])})"):
                    for s in m["sources"]:
                        st.markdown(f"- 📄 **{clean_source(s)}** · [View RBI portal]({SYNC_URL})")

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

            if idx == last_idx:
                fcols = st.columns(len(FOLLOW_UPS))
                for col, f in zip(fcols, FOLLOW_UPS):
                    with col:
                        st.button(f, key=f"fu_{idx}_{f}", on_click=ask, args=(f,))

# ──────────────────────────────────────────────────────────────
# Chat input & execution
# ──────────────────────────────────────────────────────────────
typed = st.chat_input("Ask about RBI guidelines, KYC rules, loan rates, NPAs, digital payments…")
if typed:
    ask(typed)

pending = st.session_state.pending
if pending:
    st.session_state.pending = None
    prompt = pending["text"]
    stamp = f"{datetime.now():%d %b, %H:%M}"

    # History is taken BEFORE the new question is added, so the prompt is not duplicated.
    if pending["regen"]:
        history = st.session_state.messages[:-1][-4:]
    else:
        history = st.session_state.messages[-4:]
        st.session_state.messages.append({"role": "user", "content": prompt, "ts": stamp})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏦"):
        try:
            start = time.time()
            with st.spinner(f"Searching top {top_k} RBI documents and writing your answer…"):
                history_text = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history)
                payload = (
                    f"Recent Conversation History:\n{history_text}\n\n"
                    f"Current User Request: {prompt}{STYLES[style]}"
                )
                raw = rag_engine.generate_answer(payload, top_k=top_k)
            elapsed = time.time() - start
            answer, sources = normalize(raw)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources,
                 "elapsed": elapsed, "ts": f"{datetime.now():%d %b, %H:%M}"}
            )
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate an answer: {e}")
            st.button("↻  Try again", key="retry", on_click=ask, args=(prompt, True))

if not st.session_state.messages:
    st.markdown(
        '<div class="disclaimer">BharatBank RAG is a compliance research tool. AI-generated answers should be verified against original RBI circulars.</div>',
        unsafe_allow_html=True,
    )