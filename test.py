from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=25, num_beams=10, no_repeat_ngram_size=2, top_k=100, top_p=0.85, temperature=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def chat_with_model(model, tokenizer):
    print("GPT-2 Chatbot: Enter 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        response = generate_response(user_input, model, tokenizer)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    # Load the model and tokenizer
    # model_path = "./gpt2-finetuned"  # Replace with the actual path where the model was saved
    # model = GPT2LMHeadModel.from_pretrained(model_path)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Load the fine-tuned model and tokenizer
    fine_tuned_model_path = "./gpt2-finetuned"
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_path)

    # Add a custom end-of-text token
    end_of_text_token = "<|endoftext|>"
    tokenizer.add_tokens([end_of_text_token])
    fine_tuned_model.resize_token_embeddings(len(tokenizer))

    # Test the chatbot
    chat_with_model(fine_tuned_model, tokenizer)

    queries = []