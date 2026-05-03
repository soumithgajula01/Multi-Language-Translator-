import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

# Page config
st.set_page_config(page_title="Translator", page_icon="🌍", layout="wide")

# 🎨 UI Styling
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #020617);
}
.block-container {
    padding-top: 2rem;
    max-width: 900px;
    margin: auto;
}

/* Glass effect */
section[data-testid="stVerticalBlock"] > div {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Text area */
textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Buttons */
button {
    border-radius: 10px !important;
    height: 45px;
    font-weight: 600;
}

/* Center headings */
h1, h2, h3 {
    text-align: center;
}

/* Mobile */
@media (max-width: 768px) {
    textarea {
        height: 150px !important;
    }
    button {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌍 Multi-Language Translator</h1>", unsafe_allow_html=True)
st.caption("Translate text instantly using AI")
st.divider()

# Languages
languages = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es"
}

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "text" not in st.session_state:
    st.session_state.text = ""

if "result" not in st.session_state:
    st.session_state.result = ""

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"

if "target_lang" not in st.session_state:
    st.session_state.target_lang = "French"

# 🔁 Language + Swap Layout (centered)
col1, col2, col3 = st.columns([4,1,4])

with col1:
    source_lang = st.selectbox(
        "🌐 Source",
        list(languages.keys()),
        key="source_lang"
    )

with col2:
    swap_btn = st.button("↔", use_container_width=True)

with col3:
    target_lang = st.selectbox(
        "🎯 Target",
        list(languages.keys()),
        key="target_lang"
    )

# 🔁 Swap logic (SAFE)
if swap_btn:
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang
    )
    st.session_state.text = st.session_state.result
    st.session_state.result = ""
    st.rerun()

# Input
st.session_state.text = st.text_area(
    "✏️ Enter text:",
    value=st.session_state.text,
    height=120
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    translate_btn = st.button("Translate", use_container_width=True)

with col2:
    clear_btn = st.button("Clear", use_container_width=True)

# Clear
if clear_btn:
    st.session_state.text = ""
    st.session_state.result = ""
    st.rerun()

# Model cache
@st.cache_resource
def load_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Translate function
def translate(text, src, tgt):
    tokenizer, model = load_model(src, tgt)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translate
if translate_btn:
    if st.session_state.text.strip() == "":
        st.warning("⚠️ Please enter some text")
    elif source_lang == target_lang:
        st.warning("⚠️ Source and target must differ")
    else:
        src = languages[source_lang]
        tgt = languages[target_lang]

        with st.spinner("Translating... ⏳"):
            try:
                result = translate(st.session_state.text, src, tgt)
            except:
                if src != "en" and tgt != "en":
                    step1 = translate(st.session_state.text, src, "en")
                    result = translate(step1, "en", tgt)
                else:
                    st.error("❌ Translation not supported")
                    result = None

        if result:
            st.session_state.result = result

            if not st.session_state.history or st.session_state.history[-1]["input"] != st.session_state.text:
                st.session_state.history.append({
                    "input": f"{source_lang}: {st.session_state.text}",
                    "output": f"{target_lang}: {result}"
                })

            st.session_state.history = st.session_state.history[-10:]

# Output
if st.session_state.result:
    st.markdown("### 🌐 Translated Text")
    st.success(st.session_state.result)

# History
st.divider()
st.subheader("🧠 Translation History")

if len(st.session_state.history) == 0:
    st.write("No translations yet.")
else:
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**{item['input']}**")
        st.markdown(f"➡️ {item['output']}")
        st.markdown("---")

    df = pd.DataFrame(st.session_state.history)

    st.download_button(
        label="📥 Download History",
        data=df.to_csv(index=False),
        file_name="translation_history.csv",
        mime="text/csv"
    )