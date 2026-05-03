import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

model_name = "t5-base"

# Load model once
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Page config
st.set_page_config(page_title="Translator", page_icon="🌍", layout="centered")

# Session state
if "text" not in st.session_state:
    st.session_state.text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title("🌍 English → French Translator")
st.markdown("---")

# Input (linked to session state properly)
st.session_state.text = st.text_area("✏️ Enter English text:", value=st.session_state.text)

# Buttons (aligned properly)
col1, col2 = st.columns([1, 1])

with col1:
    translate_btn = st.button("Translate", use_container_width=True)

with col2:
    clear_btn = st.button("Clear", use_container_width=True)

# ✅ Clear fix
if clear_btn:
    st.session_state.text = ""
    st.rerun()

# Translate
if translate_btn:
    if st.session_state.text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        input_text = "translate English to French: " + st.session_state.text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with st.spinner("Translating... ⏳"):
            outputs = model.generate(
                input_ids,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("### 🇫🇷 Translated Text")
        st.code(result)

        # Save history
        st.session_state.history.append({
            "input": st.session_state.text,
            "output": result
        })

# History
st.markdown("---")
st.subheader("🧠 Translation History")

if len(st.session_state.history) == 0:
    st.write("No translations yet.")
else:
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**Input:** {item['input']}")
        st.markdown(f"**Output:** {item['output']}")
        st.markdown("---")

    # Download CSV
    df = pd.DataFrame(st.session_state.history)

    st.download_button(
        label="📥 Download History",
        data=df.to_csv(index=False),
        file_name="translation_history.csv",
        mime="text/csv"
    )