import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

# Page config
st.set_page_config(page_title="Translator", page_icon="🌍", layout="centered")

st.title("🌍 Multi-Language Translator")
st.markdown("---")

# Supported languages
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

# Model cache
@st.cache_resource
def load_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Dropdowns
col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox("Source Language", list(languages.keys()))

with col2:
    target_lang = st.selectbox("Target Language", list(languages.keys()))

# 🔥 Input + Paste button layout
col1, col2 = st.columns([5,1])

with col1:
    text = st.text_area("✏️ Enter text:", value=st.session_state.text)

with col2:
    paste_btn = st.button("📋 Paste")

# Paste info (browser limitation)
if paste_btn:
    st.info("Use Ctrl+V (browser security restricts direct paste)")

# Buttons
col1, col2 = st.columns(2)

with col1:
    translate_btn = st.button("Translate", use_container_width=True)

with col2:
    clear_btn = st.button("Clear", use_container_width=True)

# ✅ FIXED Clear
if clear_btn:
    st.session_state.text = ""
    st.rerun()

# Translation function
def translate(text, src, tgt):
    tokenizer, model = load_model(src, tgt)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translate logic
if translate_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    elif source_lang == target_lang:
        st.warning("⚠️ Source and target languages must be different")
    else:
        # store text
        st.session_state.text = text

        src = languages[source_lang]
        tgt = languages[target_lang]

        with st.spinner("Translating... ⏳"):
            try:
                result = translate(text, src, tgt)
            except:
                # Pivot via English
                if src != "en" and tgt != "en":
                    step1 = translate(text, src, "en")
                    result = translate(step1, "en", tgt)
                else:
                    st.error("❌ Translation not supported")
                    result = None

        if result:
            st.markdown("### 🌐 Translated Text")
            st.code(result)

            # ✅ Prevent duplicate history
            if not st.session_state.history or st.session_state.history[-1]["input"] != text:
                st.session_state.history.append({
                    "input": f"{source_lang}: {text}",
                    "output": f"{target_lang}: {result}"
                })

            # limit history
            st.session_state.history = st.session_state.history[-10:]

# History
st.markdown("---")
st.subheader("🧠 Translation History")

if len(st.session_state.history) == 0:
    st.write("No translations yet.")
else:
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**{item['input']}**")
        st.markdown(f"➡️ {item['output']}")
        st.markdown("---")

    # Download CSV
    df = pd.DataFrame(st.session_state.history)

    st.download_button(
        label="📥 Download History",
        data=df.to_csv(index=False),
        file_name="translation_history.csv",
        mime="text/csv"
    )