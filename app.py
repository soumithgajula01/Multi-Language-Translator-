from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-base"

print("Loading model... please wait ⏳")

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("✅ Model loaded successfully!\n")

while True:
    text = input("Enter text (or type 'exit' to quit): ")

    if text.lower() == "exit":
        print("👋 Exiting translator. Goodbye!")
        break

    input_text = "translate English to French: " + text

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(
        input_ids,
        max_length=50,
        num_beams=4,
        early_stopping=True
    )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Translated Text:", translated)
    print("-" * 50)