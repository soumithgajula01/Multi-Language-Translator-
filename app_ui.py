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

# Input
text = st.text_area("✏️ Enter text:")

# Buttons
col1, col2 = st.columns(2)

with col1:
    translate_btn = st.button("Translate", use_container_width=True)

with col2:
    clear_btn = st.button("Clear", use_container_width=True)

# Clear
if clear_btn:
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
        src = languages[source_lang]
        tgt = languages[target_lang]

        with st.spinner("Translating... ⏳"):
            # Direct translation if available
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

            # Save history
            st.session_state.history.append({
                "input": f"{source_lang}: {text}",
                "output": f"{target_lang}: {result}"
            })

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