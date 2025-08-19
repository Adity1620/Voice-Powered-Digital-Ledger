import os
import io
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import soundfile as sf

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.sql import text as sql_text

# Audio recorder component
# pip install streamlit-audiorec
from st_audiorec import st_audiorec

# ==============================
# Config
# ==============================
DB_URL = "sqlite:///./voice_ledger.sqlite3"
ENABLE_WHISPER = True  # Set True to enable server-side Whisper transcription (requires ffmpeg + openai-whisper)

# ==============================
# Database setup (SQLite)
# ==============================
Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class LedgerEntry(Base):
    __tablename__ = "ledger_entries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    tx_type = Column(String(10), nullable=False)  # "income" | "expense"
    amount = Column(Float, nullable=False)
    currency = Column(String(5), default="INR")
    item = Column(String(120), nullable=True)
    category = Column(String(80), nullable=True)
    note = Column(Text, nullable=True)
    tx_date = Column(DateTime(timezone=True), server_default=func.now())
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

Base.metadata.create_all(engine)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Krishi-Mitra AI", page_icon="🎙️", layout="centered")
st.title("🎙️ Krishi-Mitra AI Voice‑Powered Hisab‑Kitab")

# Session state for parsed entry
if "parsed" not in st.session_state:
    st.session_state.parsed = None

# ==============================
# Parsing helpers (regex + keywords)
# ==============================
CATEGORIES = {
    "diesel": ("expense", "Fuel"), "fuel": ("expense", "Fuel"),
    "fertilizer": ("expense", "Fertilizer"), "urea": ("expense", "Fertilizer"),
    "seed": ("expense", "Seeds"), "seeds": ("expense", "Seeds"),
    "pesticide": ("expense", "Pesticide"), "spray": ("expense", "Pesticide"),
    "transport": ("expense", "Transport"), "tractor": ("expense", "Machinery"),
    "sold": ("income", "Crop_Sale"), "sale": ("income", "Crop_Sale"),
    "becha": ("income", "Crop_Sale"), "बेचा": ("income", "Crop_Sale"),
    "tomato": ("income", "Crop_Sale"), "onion": ("income", "Crop_Sale"),
    "paddy": ("income", "Crop_Sale"), "wheat": ("income", "Crop_Sale"), "rice": ("income", "Crop_Sale"),
}
AMOUNT_PATTERNS = [
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:rs|inr|rupees?|रुपये|₹)",
    r"(?:rs|inr|rupees?|रुपये|₹)\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:amount|amt)"
]
DATE_KEYWORDS = {"today": 0, "yesterday": 1, "आज": 0, "कल": 1}

def _extract_amount(text: str) -> Optional[float]:
    for p in AMOUNT_PATTERNS:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            amt = m.group(1).replace(",", "")
            try:
                return float(amt)
            except:
                continue
    m = re.search(r"\b(\d{2,7})\b", text)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def _infer_type_and_category(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    t = text.lower()
    for k, (tx_type, cat) in CATEGORIES.items():
        if k in t:
            return tx_type, k, cat
    if any(x in t for x in ["बेचा","sold","sale","received","मिले"]):
        return "income", None, "Crop_Sale"
    if any(x in t for x in ["खरीदा","buy","bought","spent","खर्च"]):
        return "expense", None, "Other"
    return "expense", None, "Other"

def _extract_date(text: str) -> Optional[datetime]:
    t = text.lower()
    for k, d in DATE_KEYWORDS.items():
        if k in t:
            return datetime.now() - timedelta(days=d)
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", t)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return datetime(year=y, month=mth, day=d)
        except:
            return None
    return None

def parse_text_transaction(text_value: str, stt_conf: float = 0.0) -> Dict[str, Any]:
    amount = _extract_amount(text_value) or 0.0
    tx_type, item_kw, category = _infer_type_and_category(text_value)
    when = _extract_date(text_value) or datetime.now()
    conf = min(1.0, max(0.1, stt_conf) + (0.2 if amount > 0 else 0.0))
    return {
        "tx_type": tx_type,
        "amount": amount,
        "item": item_kw,
        "category": category,
        "note": None,
        "tx_date": when.isoformat(),
        "confidence": conf,
        "raw_text": text_value
    }

# ==============================
# Optional: Whisper transcription
# ==============================
def transcribe_whisper_from_bytes(wav_bytes: bytes, lang_hint: Optional[str] = None) -> str:
    if not ENABLE_WHISPER:
        return ""
    try:
        import whisper
    except Exception:
        st.error("Whisper not installed. Install with: pip install openai-whisper (and ensure ffmpeg is available).")
        return ""
    try:
        # Save to temp WAV file first
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        model = whisper.load_model("base")  # or "small"
        result = model.transcribe(tmp_path, language=lang_hint) if lang_hint else model.transcribe(tmp_path)
        return (result.get("text") or "").strip()
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return ""

# ==============================
# Record (streamlit-audiorec)
# ==============================
st.subheader("🎤 Record (Browser Mic)")
st.caption("Click the mic button, speak for 5–10s, then stop. The recorder returns WAV bytes directly.")

wav_audio_bytes = st_audiorec()  # returns WAV bytes or None

# Process recording
st.subheader("🛠️ Process Recording")
if st.button("Process Recording"):
    if wav_audio_bytes is None:
        st.error("No recording found. Please record first.")
    else:
        try:
            # Validate/normalize audio using soundfile
            data, sr = sf.read(io.BytesIO(wav_audio_bytes), dtype="float32", always_2d=False)
            if isinstance(data, np.ndarray) and data.ndim > 1:
                data = data.mean(axis=1)  # mono
            # Re-encode to a clean WAV buffer
            buf = io.BytesIO()
            sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
            wav_clean_bytes = buf.getvalue()
            st.success(f"Audio captured (sr={sr} Hz).")

            transcript = ""
            if ENABLE_WHISPER:
                with st.spinner("Transcribing with Whisper..."):
                    transcript = transcribe_whisper_from_bytes(wav_clean_bytes, None)

            if transcript:
                st.write("Transcript:")
                st.write(transcript)
                parsed = parse_text_transaction(transcript, 0.9)
                st.session_state.parsed = parsed
                st.success("Parsed transcript into a transaction.")
                with st.expander("Parsed fields"):
                    st.json(parsed)
            else:
                st.info("Transcription disabled or not used. Type below to parse manually.")

        except Exception as e:
            st.error(f"Audio processing failed: {e}")

# ==============================
# Text input (manual)
# ==============================
st.divider()
st.subheader("💬 Or type a sentence")
typed = st.text_input("e.g., 'Diesel 800 rupees' or 'Sold onion 4200 today'")
if st.button("Parse Text"):
    if typed.strip():
        st.session_state.parsed = parse_text_transaction(typed.strip(), 0.9)
        st.success("Parsed text input")
        with st.expander("Parsed fields"):
            st.json(st.session_state.parsed)

# ==============================
# Confirm & Save
# ==============================
db = SessionLocal()
st.subheader("✅ Confirm & Save")
if st.session_state.parsed:
    p = st.session_state.parsed
    tx_type = st.selectbox("Type", ["income","expense"], index=0 if p["tx_type"]=="income" else 1)
    amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, value=float(p["amount"]))
    item = st.text_input("Item", value=p.get("item") or "")
    category = st.text_input("Category", value=p.get("category") or "")
    note = st.text_area("Note", value=p.get("raw_text") or "", height=70)
    try:
        dflt = datetime.fromisoformat(p["tx_date"]).date()
    except Exception:
        dflt = datetime.now().date()
    tx_date = st.date_input("Date", value=dflt)

    if st.button("Save Entry"):
        try:
            entry = LedgerEntry(
                tx_type=tx_type,
                amount=amount,
                item=item or None,
                category=category or None,
                note=note or None,
                tx_date=datetime.combine(tx_date, datetime.min.time()),
                meta={"raw": p}
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            st.success(f"Saved #{entry.id}")
            st.session_state.parsed = None
        except Exception as e:
            st.error(f"Save failed: {e}")

# ==============================
# Recent Entries
# ==============================
st.subheader("📄 Recent Entries")
try:
    rows = db.execute(sql_text("""
        SELECT id, tx_date, tx_type, amount, item, category, note, created_at
        FROM ledger_entries
        ORDER BY tx_date DESC, id DESC
        LIMIT 50
    """)).fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=["id","tx_date","tx_type","amount","item","category","note","created_at"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export CSV", data=df.to_csv(index=False), file_name="ledger_recent.csv", mime="text/csv")
    else:
        st.info("No entries yet.")
except Exception as e:
    st.error(f"Load failed: {e}")

# ==============================
# Summary Cards
# ==============================
st.subheader("📌 Summary")
def period_summary(sql_where: str) -> Tuple[float, float]:
    qrow = db.execute(sql_text(f"""
        SELECT 
          COALESCE(SUM(CASE WHEN tx_type='income' THEN amount END),0) as inc,
          COALESCE(SUM(CASE WHEN tx_type='expense' THEN amount END),0) as exp
        FROM ledger_entries
        WHERE {sql_where}
    """)).fetchone()
    income = float(qrow[0] or 0.0)
    expense = float(qrow[1] or 0.0)
    return income, expense

cols = st.columns(4)
periods = {
    "TODAY": "date(tx_date) = date('now','localtime')",
    "7D": "tx_date >= datetime('now','-7 day','localtime')",
    "30D": "tx_date >= datetime('now','-30 day','localtime')",
    "ALL": "1=1"
}
for (label, wh), c in zip(periods.items(), cols):
    inc, exp = period_summary(wh)
    with c:
        st.metric(label, value=f"₹{inc-exp:.0f}", delta=f"Inc ₹{inc:.0f} | Exp ₹{exp:.0f}")

db.close()
