from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned GPT-2 model and tokenizer
model_path = "./gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Tokenize the user input
        input_ids = tokenizer.encode(user_input, return_tensors="pt")

        # Generate response
        output = model.generate(input_ids, max_length=25, num_beams=10, no_repeat_ngram_size=2, top_k=100, top_p=0.85, temperature=0.9)
        generated_reply = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the initial part of the bot's response if it's the same as the input
        if generated_reply.startswith(user_input):
            generated_reply = generated_reply[len(user_input):].lstrip()

        # Truncate the response after ".", ",", "!"
        stop_chars = [".", "?", "!"]
        for char in stop_chars:
            if char in generated_reply:
                generated_reply = generated_reply.split(char, 1)[0]
                break

        return jsonify({'response': generated_reply})

if __name__ == '__main__':
    app.run(debug=True)
