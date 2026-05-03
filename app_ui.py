import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-base"

# Load model only once (important)
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# UI Design
st.set_page_config(page_title="Translator", page_icon="🌍")

st.title("🌍 English → French Translator")
st.write("Translate English text into French using AI")

# Input box
text = st.text_area("Enter text:")

# Translate button
if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        input_text = "translate English to French: " + text
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        outputs = model.generate(
            input_ids,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.success("Translated Text:")
        st.write(result)