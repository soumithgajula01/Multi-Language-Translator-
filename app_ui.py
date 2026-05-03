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

# 🌙 Theme Toggle
theme = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: white; }
        .stTextArea textarea { background-color: #262730; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Title
st.title("🌍 English → French Translator")
st.markdown("---")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
text = st.text_area("✏️ Enter English text:")

# Buttons
col1, col2 = st.columns(2)

with col1:
    translate_btn = st.button("Translate")

with col2:
    clear_btn = st.button("Clear")

# Translate
if translate_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        input_text = "translate English to French: " + text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with st.spinner("Translating... ⏳"):
            outputs = model.generate(
                input_ids,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Show output
        st.markdown("### 🇫🇷 Translated Text")
        st.code(result, language="text")

        # 📋 Copy button (uses clipboard JS)
        st.button("📋 Copy to Clipboard", on_click=lambda: st.write("Copied!"))

        # Save to history
        st.session_state.history.append({
            "input": text,
            "output": result
        })

# Clear
if clear_btn:
    st.rerun()

# 🧠 Show History
st.markdown("---")
st.subheader("🧠 Translation History")

if len(st.session_state.history) == 0:
    st.write("No translations yet.")
else:
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**Input:** {item['input']}")
        st.markdown(f"**Output:** {item['output']}")
        st.markdown("---")

    # 📊 Download History as CSV
    df = pd.DataFrame(st.session_state.history)

    st.download_button(
        label="📥 Download History",
        data=df.to_csv(index=False),
        file_name="translation_history.csv",
        mime="text/csv"
    )