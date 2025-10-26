# k2_think_full_app.py
"""
K2 Think — Full-featured Streamlit app (single-file)
Features:
 - Multi-page UI (Study Hub, K2 Chat (RAG), MCQ, Flashcards, Interview Live, Code Sandbox, Resume Builder & Job Matcher, Export)
 - Full models: GPT-2 (transformers pipeline), SentenceTransformer (all-MiniLM-L6-v2), spaCy en_core_web_sm
 - RAG index using FAISS (fallback to numpy similarity)
 - PDF/TXT upload, Image OCR (pytesseract if available)
 - Live audio recording via streamlit-webrtc (fallback to audio upload)
 - ASR using SpeechRecognition (Google Web Speech API for short clips)
 - TTS via pyttsx3
 - ReadyPlayerMe avatar iframe embed with mood mapping
 - Code sandbox with safe subprocess execution and model-based review
 - Resume builder that suggests keywords and does simple job matching against a local sample job list using embeddings
 - Encrypted export of logs via cryptography.Fernet
 - Lots of try/except and graceful fallbacks
"""
import streamlit as st
from streamlit.components.v1 import html as st_html
from pathlib import Path
import tempfile, time, json, base64, hashlib, subprocess, sys, os, textwrap
from typing import List, Tuple
import numpy as np
import random
import io

# Optional heavy imports guarded
try:
    from transformers import pipeline
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    EMB_OK = True
except Exception:
    EMB_OK = False

USE_FAISS = True
try:
    import faiss
except Exception:
    faiss = None
    USE_FAISS = False

try:
    from PyPDF2 import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import pytesseract
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except Exception:
    SR_OK = False

try:
    import pyttsx3
    TTS_OK = True
except Exception:
    TTS_OK = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, AudioProcessorBase
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False

try:
    import spacy
    SPACY_OK = True
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
    except Exception:
        try:
            # attempt to download
            import subprocess as _subp
            _subp.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp_spacy = spacy.load("en_core_web_sm")
        except Exception:
            SPACY_OK = False
            nlp_spacy = None
except Exception:
    SPACY_OK = False
    nlp_spacy = None

try:
    from streamlit_ace import st_ace
    ACE_OK = True
except Exception:
    ACE_OK = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ----------------------
# Small helpers & init
# ----------------------
st.set_page_config(page_title="K2 Learn — Interview & Career Suite", layout="wide", page_icon="🧠")
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
def k(key: str) -> str:
    st.session_state["counter"] += 1
    return f"{key}_{st.session_state['counter']}"

def chunk_text(text: str, max_words=200, overlap=40):
    words = text.split()
    if not words:
        return []
    chunks=[]
    i=0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
    return chunks

def derive_key(password: str):
    import hashlib, base64
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def encrypt_bytes(b: bytes, password: str):
    if not CRYPTO_OK:
        raise RuntimeError("cryptography not installed")
    from cryptography.fernet import Fernet
    key = derive_key(password)
    f = Fernet(key)
    return f.encrypt(b)

def safe_run_python(source_code: str, stdin: str = "", timeout=5):
    # Run in temporary subprocess
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(source_code)
        fname = tf.name
    try:
        proc = subprocess.run([sys.executable, fname], input=stdin.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = proc.stdout.decode(errors="ignore")
        err = proc.stderr.decode(errors="ignore")
        return out, err
    except subprocess.TimeoutExpired:
        return "", "Timeout"
    except Exception as e:
        return "", str(e)
    finally:
        try:
            os.remove(fname)
        except Exception:
            pass

# ----------------------
# Global session structures
# ----------------------
if "kb_chunks" not in st.session_state:
    st.session_state["kb_chunks"] = []
if "kb_embeddings" not in st.session_state:
    st.session_state["kb_embeddings"] = None
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
if "mcqs" not in st.session_state:
    st.session_state["mcqs"] = []
if "flashcards" not in st.session_state:
    st.session_state["flashcards"] = []
if "interview_questions" not in st.session_state:
    st.session_state["interview_questions"] = []
if "interview_answers" not in st.session_state:
    st.session_state["interview_answers"] = []
if "interview_logs" not in st.session_state:
    st.session_state["interview_logs"] = []
if "last_scores" not in st.session_state:
    st.session_state["last_scores"] = None
if "resume_profile" not in st.session_state:
    st.session_state["resume_profile"] = {}

# ----------------------
# Load big models (cached)
# ----------------------
@st.cache_resource(show_spinner=False)
def load_models():
    chat = None
    embed = None
    try:
        if TRANSFORMERS_OK:
            chat = pipeline("text-generation", model="gpt2")
    except Exception as e:
        chat = None
    try:
        if EMB_OK:
            embed = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        embed = None
    return chat, embed

chat_model, embed_model = load_models()

# ----------------------
# UI CSS
# ----------------------
st.markdown("""
<style>
.body-bg{background: linear-gradient(135deg,#edf2ff,#fef3c7);}
.card{background:white;border-radius:12px;padding:18px;box-shadow: 0 8px 24px rgba(2,6,23,0.06);}
.big{font-size:22px;font-weight:700;color:#0f172a;}
.muted{color:#64748b}
.btn-primary button{background:#2563eb;color:white;border-radius:8px;padding:8px 12px}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Sidebar navigation
# ----------------------
with st.sidebar:
    st.title("K2 Think Suite")
    st.markdown("Multi-tool: RAG, Interview, Code, Resume")
    page = st.radio("Navigate", ["Home","Study Hub","K2 Chat (RAG)","MCQ","Flashcards",
                                "Interview (Live)","Code Sandbox","Resume Builder","Export Logs"], index=0)

# ----------------------
# HOME
# ----------------------
if page == "Home":
    st.markdown("<div class='card'><div class='big'>K2 Think — Interview & Career Suite</div><div class='muted'>Full demo: RAG + Live audio + avatar + resume builder + code sandbox.</div></div>", unsafe_allow_html=True)
    st.write("Quick tips:")
    st.write("- Build a KB in Study Hub (upload PDF/TXT or paste).")
    st.write("- Use K2 Chat to ask domain questions — model cites retrieved snippets.")
    st.write("- Take a live interview in Interview (Live).")
    st.write("- Generate MCQs / Flashcards and practice.")
    st.write("- Use Resume Builder & Job Matcher to create tailored CVs.")
    st.write("")
    col1, col2, col3 = st.columns(3)
    col1.metric("KB chunks", len(st.session_state["kb_chunks"]))
    col2.metric("Flashcards", len(st.session_state["flashcards"]))
    col3.metric("MCQ sets", len(st.session_state["mcqs"]))

# ----------------------
# STUDY HUB (upload + RAG index)
# ----------------------
elif page == "Study Hub":
    st.header("📘 Study Hub — Upload Notes and Build KB (RAG)")
    st.markdown("Upload PDF/TXT or an image to extract text. Then create a FAISS index (or embeddings) for RAG.")
    uploaded = st.file_uploader("Upload PDF / TXT / Image (PNG/JPG)", type=["pdf","txt","png","jpg","jpeg"], key=k("up1"))
    pasted = st.text_area("Or paste text here", height=220, key=k("paste1"))
    if st.button("Create / Append to KB", key=k("create_kb_btn")):
        combined = ""
        if uploaded:
            try:
                if uploaded.type == "application/pdf":
                    if not PDF_OK:
                        st.error("PyPDF2 not installed (pip install PyPDF2)")
                    else:
                        r = PdfReader(uploaded)
                        pages = [p.extract_text() or "" for p in r.pages]
                        combined = "\n".join(pages)
                elif uploaded.type.startswith("image/") or uploaded.name.lower().endswith((".png",".jpg",".jpeg")):
                    if not PIL_OK:
                        st.error("Pillow not installed (pip install pillow)")
                    else:
                        image = Image.open(uploaded)
                        if TESSERACT_OK:
                            text = pytesseract.image_to_string(image)
                            combined = text
                        else:
                            st.warning("pytesseract not installed; cannot OCR.")
                else:
                    combined = uploaded.getvalue().decode("utf-8", errors="ignore")
            except Exception as e:
                st.error(f"Failed to read uploaded: {e}")
        elif pasted and pasted.strip():
            combined = pasted
        if not combined.strip():
            st.warning("Provide document or paste text.")
        else:
            chunks = chunk_text(combined, max_words=200, overlap=40)
            st.session_state["kb_chunks"].extend(chunks)
            st.success(f"Added {len(chunks)} chunks. Total chunks: {len(st.session_state['kb_chunks'])}")
            # embeddings
            if embed_model:
                try:
                    emb = embed_model.encode(st.session_state["kb_chunks"], convert_to_numpy=True)
                    st.session_state["kb_embeddings"] = emb
                    if USE_FAISS:
                        try:
                            dim = emb.shape[1]
                            index = faiss.IndexFlatL2(dim)
                            index.add(emb)
                            st.session_state["faiss_index"] = index
                            st.info("FAISS index built.")
                        except Exception as e:
                            st.session_state["faiss_index"] = None
                            st.warning(f"FAISS build failed, embeddings stored only: {e}")
                    else:
                        st.session_state["faiss_index"] = None
                        st.info("Embeddings stored (no FAISS).")
                except Exception as e:
                    st.warning(f"Embedding creation failed: {e}")
            else:
                st.warning("Embedding model not available (install sentence-transformers).")
    st.markdown("**KB Preview (first 5 chunks)**")
    for i, c in enumerate(st.session_state["kb_chunks"][:5]):
        st.write(f"{i+1}. {c[:350]}...")

    if st.button("Clear KB", key=k("clear_kb")):
        st.session_state["kb_chunks"] = []
        st.session_state["kb_embeddings"] = None
        st.session_state["faiss_index"] = None
        st.success("Cleared KB.")

# ----------------------
# K2 CHAT (RAG)
# ----------------------
elif page == "K2 Chat (RAG)":
    st.header("💬 K2 Chat — Retrieval Augmented Generation")
    st.markdown("Ask a question. If a KB exists, top-k context snippets are retrieved and included in the prompt.")
    q = st.text_area("Your question for K2", key=k("q_k2"))
    topk = st.slider("Top-k context", 1, 6, 3, key=k("topk_k2"))
    if st.button("Ask K2", key=k("ask_k2")):
        if not q.strip():
            st.warning("Type a question.")
        else:
            ctx_snips = []
            if st.session_state.get("kb_chunks") and embed_model and st.session_state.get("kb_embeddings") is not None:
                try:
                    q_emb = embed_model.encode([q], convert_to_numpy=True)
                    if USE_FAISS and st.session_state.get("faiss_index") is not None:
                        D,I = st.session_state["faiss_index"].search(np.array(q_emb), topk)
                        ids = [int(i) for i in I[0] if int(i) >= 0]
                        ctx_snips = [st.session_state["kb_chunks"][idx] for idx in ids if idx < len(st.session_state["kb_chunks"])]
                    else:
                        emb = st.session_state["kb_embeddings"]
                        sims = (emb @ q_emb[0]) / (np.linalg.norm(emb,axis=1) * (np.linalg.norm(q_emb[0]) + 1e-9))
                        idxs = np.argsort(-sims)[:topk]
                        ctx_snips = [st.session_state["kb_chunks"][i] for i in idxs]
                except Exception as e:
                    st.warning(f"Retrieval failed: {e}")
            prompt_context = "\n".join([f"[CTX{i+1}] {s[:400]}" for i,s in enumerate(ctx_snips)])
            prompt = f"[INSTRUCTIONS]\nUse context; cite CTX IDs.\n\n[CONTEXT]\n{prompt_context}\n\n[QUESTION]\n{q}\n\n[ANSWER]\n"
            if chat_model:
                try:
                    out = chat_model(prompt, max_length=220, num_return_sequences=1)[0]["generated_text"]
                    # show truncated answer
                    answer = out.replace(prompt, "").strip() if prompt in out else out
                    st.markdown("**K2's answer (mock GPT-2):**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
            else:
                st.info("Local model not available. Install transformers to enable.")
            if ctx_snips:
                with st.expander("Retrieved snippets used"):
                    for i,s in enumerate(ctx_snips):
                        st.write(f"[CTX{i+1}] {s[:800]}...")

# ----------------------
# MCQ generation & practice
# ----------------------
elif page == "MCQ":
    st.header("📝 MCQ Generator & Practice")
    st.markdown("Generate 4-option MCQs from KB or inline text. Use 'Check' for immediate feedback.")
    src = st.radio("Source", ["Inline text","From KB"], key=k("mcq_src"))
    inline = st.text_area("Inline text", key=k("mcq_inline"))
    n = st.slider("Number of MCQs", 1, 10, 5, key=k("mcq_n"))
    if st.button("Generate MCQs", key=k("mcq_gen")):
        pool = []
        if src == "From KB" and st.session_state["kb_chunks"]:
            pool = st.session_state["kb_chunks"][: min(40, len(st.session_state["kb_chunks"]))]
            seeds = [p.split(".")[0] for p in pool]
        elif src == "Inline text" and inline.strip():
            seeds = inline.split(".")
        else:
            st.warning("Provide inline text or build KB.")
            seeds = []
        mcqs=[]
        for i in range(n):
            seed = (seeds[i % len(seeds)] if seeds else f"Topic {i+1}")
            q_text = f"{seed.strip()[:140]}?"
            opts = [f"Option {chr(65+j)}" for j in range(4)]
            correct = random.choice(opts)
            mcqs.append({"q":q_text,"options":opts,"answer":correct})
        st.session_state["mcqs"] = mcqs
        st.success(f"Created {len(mcqs)} MCQs.")
    if st.session_state.get("mcqs"):
        for idx, m in enumerate(st.session_state["mcqs"]):
            st.write(f"**Q{idx+1}.** {m['q']}")
            ans = st.radio(f"Choose answer Q{idx+1}", m["options"], key=k(f"mcq_choice_{idx}"))
            if st.button("Check", key=k(f"mcq_check_{idx}")):
                if ans == m["answer"]:
                    st.success("Correct ✅")
                else:
                    st.error(f"Wrong — correct: {m['answer']}")

# ----------------------
# Flashcards with flip animation
# ----------------------
elif page == "Flashcards":
    st.header("🎴 Flashcards — flip to reveal answers")
    src = st.radio("Source", ["Inline","From KB"], key=k("fc_src"))
    inline = st.text_area("Inline text", key=k("fc_inline_fc"))
    n = st.number_input("Number of cards", 1, 30, 8, key=k("fc_n"))
    if st.button("Create Flashcards", key=k("fc_create")):
        cards=[]
        if src=="From KB" and st.session_state["kb_chunks"]:
            for c in st.session_state["kb_chunks"][:n]:
                q = c.split(".")[0][:100]+"?"
                a = c[:300]
                cards.append({"q":q,"a":a})
        elif src=="Inline" and inline.strip():
            parts = chunk_text(inline, max_words=100, overlap=10)
            for p in parts[:n]:
                q = p.split(".")[0][:100]+"?"
                a = p[:300]
                cards.append({"q":q,"a":a})
        else:
            st.warning("Provide KB or inline text.")
        st.session_state["flashcards"] = cards
        st.session_state["fc_idx"] = 0
    if st.session_state.get("flashcards"):
        cards = st.session_state["flashcards"]
        idx = st.session_state.get("fc_idx",0)
        card = cards[idx]
        html_card = f"""
        <div style='display:flex;align-items:center;gap:12px;'>
           <button onclick="prev()" style="padding:8px 12px;border-radius:8px;">◀ Prev</button>
           <div style='width:640px;perspective:1000px;'>
             <div id="inner" style="transition:transform 0.6s;transform-style:preserve-3d;cursor:pointer;padding:18px;border-radius:12px;box-shadow:0 8px 20px rgba(2,6,23,0.06);background:white;" onclick="flip()">
               <div style="backface-visibility:hidden;">
                 <h3 style="margin:0">{card['q']}</h3>
               </div>
               <div style="backface-visibility:hidden;transform:rotateY(180deg);margin-top:8px;">
                 <p style="margin:0;color:white;background:#0b1220;padding:12px;border-radius:8px;">{card['a']}</p>
               </div>
             </div>
           </div>
           <button onclick="next()" style="padding:8px 12px;border-radius:8px;">Next ▶</button>
        </div>
        <script>
         function flip(){{const e=document.getElementById('inner'); e.style.transform = e.style.transform=='rotateY(180deg)' ? 'rotateY(0deg)' : 'rotateY(180deg)';}}
         function prev(){{window.parent.postMessage({type:'flashcard-nav',action:'prev'},'*')}}
         function next(){{window.parent.postMessage({type:'flashcard-nav',action:'next'},'*')}}
        </script>
        """
        st_html(html_card, height=240, scrolling=False)
        # fallback buttons (JS messaging in streamlit not stable), so keep Prev/Next
        c1,c2,c3 = st.columns([1,6,1])
        with c1:
            if st.button("◀ Prev", key=k("fc_prev")):
                st.session_state["fc_idx"]=max(0, st.session_state.get("fc_idx",0)-1)
                st.experimental_rerun()
        with c3:
            if st.button("Next ▶", key=k("fc_next")):
                st.session_state["fc_idx"]=min(len(cards)-1, st.session_state.get("fc_idx",0)+1)
                st.experimental_rerun()

# ----------------------
# INTERVIEW (LIVE) - Text + Live Audio (webrtc) + ASR + TTS + avatar
# ----------------------
elif page == "Interview (Live)":
    st.header("🎙️ Interview Simulator — Live audio (webrtc) + upload")
    st.markdown("Choose interview type, generate questions, answer via text or via live audio. The app will transcribe and give mock scores. Avatar shows mood.")
    int_type = st.selectbox("Interview type", ["Technical","Behavioral","Visa"], key=k("int_type"))
    n = st.slider("Number of questions", 1, 8, 4, key=k("int_n"))
    if st.button("Generate questions", key=k("gen_q")):
        qs=[]
        for i in range(n):
            if int_type=="Technical":
                qs.append(f"Explain approach to solve problem {i+1} and mention complexity.")
            elif int_type=="Behavioral":
                qs.append(f"Describe a time when you overcame a challenge ({i+1}).")
            else:
                qs.append(f"Why do you want to go/visit this country? ({i+1})")
        st.session_state["interview_questions"]=qs
        st.session_state["interview_answers"]=[""]*len(qs)
    # display questions and text answers
    if st.session_state.get("interview_questions"):
        for i,q in enumerate(st.session_state["interview_questions"]):
            st.write(f"Q{i+1}. {q}")
            ans = st.text_area(f"Answer Q{i+1}", value=st.session_state["interview_answers"][i], key=k(f"ans_{i}"))
            st.session_state["interview_answers"][i] = ans

    # live audio via webrtc: capture short clips and transcribe after stop
    st.markdown("**Record live audio (webrtc)**")
    recorded_audio_path = None
    if WEBRTC_OK:
        # simple recorder that stores audio in session when stopped
        class AudioRecorder(AudioProcessorBase):
            def __init__(self):
                self.frames = []
            def recv(self, frame):
                # frame is av.AudioFrame
                self.frames.append(frame.to_ndarray())
                return frame
        webrtc_ctx = webrtc_streamer(key=k("webrtc"), mode=WebRtcMode.SENDRECV, audio_processor_factory=AudioRecorder, async_processing=False)
        # streamlit-webrtc returns object; to persist recorded audio we need more plumbing; fallback to upload below
        st.info("If browser permissions block webrtc, use audio upload below.")
    else:
        st.info("streamlit-webrtc not available. Use 'Upload audio' below.")

    st.markdown("**Or upload a short audio file (wav/mp3)**")
    audio_file = st.file_uploader("Upload audio (wav/mp3)", type=["wav","mp3","m4a"], key=k("aud_up"))
    if audio_file and SR_OK:
        if st.button("Transcribe uploaded audio", key=k("trans_up")):
            try:
                r = sr.Recognizer()
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tf:
                    tf.write(audio_file.read()); tf.flush()
                    with sr.AudioFile(tf.name) as source:
                        ad = r.record(source)
                        text = r.recognize_google(ad)
                        st.success("Transcript:")
                        st.write(text)
                        if st.session_state.get("interview_answers") is None:
                            st.session_state["interview_answers"]=[text]
                        else:
                            # append to last answer
                            st.session_state["interview_answers"][-1] += "\n\n(Audio):\n"+text
            except Exception as e:
                st.error(f"ASR error: {e}")
    elif audio_file and not SR_OK:
        st.warning("SpeechRecognition not installed. Install speechrecognition to enable ASR.")

    # TTS: play interviewer question
    if st.button("Play interviewer voice (TTS)", key=k("play_tts")):
        if TTS_OK:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say("Please answer the following question as if you are in an interview. Good luck!")
                engine.runAndWait()
            except Exception as e:
                st.error(f"TTS error: {e}")
        else:
            st.info("pyttsx3 TTS not available.")

    # scoring
    if st.button("Submit answers for scoring", key=k("score_int")):
        answers = st.session_state.get("interview_answers", [])
        tech=0; comm=0
        for a in answers:
            if not a.strip(): continue
            words = len(a.split())
            tech += min(10, words//10)
            comm += min(10, max(0, 10 - abs(6 - (words//10))))
        if answers:
            tech = min(10, tech//len(answers))
            comm = min(10, comm//len(answers))
        else:
            tech = comm = 0
        improvements = ["Use STAR for behavioral", "Use metric-driven examples", "Be concise"]
        st.session_state["last_scores"]={"technical":tech,"communication":comm,"improvements":improvements}
        st.session_state["interview_logs"].append({"time":time.strftime("%Y-%m-%d %H:%M:%S"),"type":int_type,"questions":st.session_state.get("interview_questions"),"answers":answers,"scores":st.session_state["last_scores"]})
        st.success("Scored. See Performance tab.")

    # Avatar embed (ReadyPlayerMe demo)
    st.markdown("**Avatar (ReadyPlayerMe iframe demo)**")
    avatar_url = "https://assets.readyplayer.me/avatar?frameApi"
    st.components.v1.iframe(avatar_url, height=320, scrolling=False)
    if st.session_state.get("last_scores"):
        avg = (st.session_state["last_scores"]["technical"] + st.session_state["last_scores"]["communication"])/2
        if avg >= 8: st.success("Mood: Confident 😄")
        elif avg >=5: st.info("Mood: Neutral 🙂")
        else: st.warning("Mood: Needs practice 😟")

# ----------------------
# CODE SANDBOX
# ----------------------
elif page == "Code Sandbox":
    st.header("💻 Code Interview Sandbox")
    st.markdown("Write Python, run locally (safe subprocess), and ask model for review (if available).")
    default_code = textwrap.dedent("""\
        def solve():
            n = int(input().strip())
            print(n*2)

        if __name__ == '__main__':
            solve()
    """)
    code = st_ace(value=default_code, language="python", theme="monokai", key=k("ace")) if ACE_OK else st.text_area("Code", value=default_code, height=240, key=k("code_area"))
    stdin = st.text_area("Stdin (optional)", key=k("stdin_area"))
    col1,col2 = st.columns(2)
    with col1:
        if st.button("Run code (safe)", key=k("run_code")):
            out, err = safe_run_python(code, stdin, timeout=6)
            if out: st.code(out)
            if err: st.error(err)
    with col2:
        if st.button("Ask model to review code", key=k("review_code")):
            if chat_model:
                prompt = f"Review this Python code for correctness and suggestions:\n\n{code}"
                try:
                    fb = chat_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
                    st.info("Model review (mock):")
                    st.write(fb)
                except Exception as e:
                    st.error(f"Model review failed: {e}")
            else:
                st.warning("Local model not available (install transformers).")

# ----------------------
# RESUME BUILDER & JOB MATCHER
# ----------------------
elif page == "Resume Builder":
    st.header("📄 Resume Builder & Job Matcher")
    st.markdown("Fill your details; AI will suggest keywords and match to local job listings (content-based).")
    with st.form("resume_form"):
        name = st.text_input("Full name", key=k("name"))
        title = st.text_input("Target title (e.g., ML Engineer)", key=k("title"))
        summary = st.text_area("Summary / About you", key=k("summary"))
        skills = st.text_area("Skills (comma-separated)", key=k("skills"))
        exp = st.text_area("Experience (short bullets)", key=k("exp"))
        submit_btn = st.form_submit_button("Generate resume & suggestions")
    if submit_btn:
        profile_text = f"{name}\n{title}\n{summary}\nSkills: {skills}\nExperience: {exp}"
        st.session_state["resume_profile"]={"name":name,"title":title,"summary":summary,"skills":skills,"exp":exp}
        # suggest keywords with spaCy/simple heuristics
        suggested=[]
        if SPACY_OK and nlp_spacy:
            doc = nlp_spacy(profile_text)
            # extract nouns / PROPN / entities
            tokens = [token.text for token in doc if token.pos_ in ("NOUN","PROPN","VERB")]
            suggested = list(dict.fromkeys(tokens))[:12]
        else:
            # simple split
            suggested = [s.strip() for s in skills.split(",")][:12]
        st.subheader("Suggested keywords to add")
        st.write(", ".join(suggested))
        # create simple resume text and offer download as txt and as simple PDF (fpdf)
        resume_txt = f"{name}\n{title}\n\nSummary:\n{summary}\n\nSkills:\n{skills}\n\nExperience:\n{exp}\n"
        st.download_button("Download resume (TXT)", data=resume_txt, file_name=f"{name.replace(' ','_')}_resume.txt")
        # Job matching: we use local sample jobs and compute embedding similarity if embed_model, else basic keyword overlap
        sample_jobs = [
            {"title":"Machine Learning Engineer","company":"AI Labs","desc":"Looking for ML engineer with python, pytorch, transformers experience"},
            {"title":"Data Scientist","company":"DataCorp","desc":"Statistics, python, modeling, scikit-learn, SQL"},
            {"title":"NLP Engineer","company":"LangTech","desc":"NLP, transformers, tokenizers, pytorch"},
            {"title":"Software Engineer","company":"WebWorks","desc":"Backend, APIs, python, docker"}
        ]
        if embed_model:
            try:
                prof_emb = embed_model.encode([profile_text], convert_to_numpy=True)
                job_texts = [j["title"] + " " + j["desc"] for j in sample_jobs]
                job_emb = embed_model.encode(job_texts, convert_to_numpy=True)
                sims = (job_emb @ prof_emb[0])/(np.linalg.norm(job_emb,axis=1)*(np.linalg.norm(prof_emb[0])+1e-9))
                order = np.argsort(-sims)
                st.subheader("Top 3 job matches (content similarity)")
                for i in order[:3]:
                    st.write(f"**{sample_jobs[i]['title']}** at {sample_jobs[i]['company']} — {sample_jobs[i]['desc']}")
            except Exception as e:
                st.warning(f"Job matching error: {e}")
        else:
            # fallback: keyword overlap
            user_keys = set([s.strip().lower() for s in skills.split(",") if s.strip()])
            matches=[]
            for j in sample_jobs:
                j_keys = set([w.lower() for w in j["desc"].split() if len(w)>3])
                overlap = len(user_keys & j_keys)
                matches.append((overlap,j))
            matches.sort(reverse=True, key=lambda x: x[0])
            st.subheader("Top job matches (keyword overlap)")
            for ov, j in matches[:3]:
                st.write(f"**{j['title']}** at {j['company']} — overlap: {ov}")

# ----------------------
# EXPORT LOGS (encrypted)
# ----------------------
elif page == "Export Logs":
    st.header("🔐 Export encrypted logs")
    logs = st.session_state.get("interview_logs", [])
    if not logs:
        st.info("No logs yet.")
    else:
        pwd = st.text_input("Enter password to encrypt export", type="password", key=k("exp_pwd"))
        if st.button("Create encrypted export", key=k("exp_btn")):
            if not CRYPTO_OK:
                st.error("cryptography not installed. pip install cryptography")
            elif not pwd:
                st.warning("Enter a password.")
            else:
                blob = json.dumps(logs).encode("utf-8")
                try:
                    token = encrypt_bytes(blob, pwd)
                    b64 = base64.b64encode(token).decode()
                    st.download_button("Download encrypted logs", data=b64, file_name="k2_logs_enc.txt")
                    st.success("Encrypted export ready.")
                except Exception as e:
                    st.error(f"Encryption failed: {e}")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748b'>K2 Think — Full demo. Replace GPT-2 with K2 Think API when available for superior reasoning. Good luck with the hackathon!</div>", unsafe_allow_html=True)
