import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Key not found")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

system_prompt = "You are a helpful AI assistant specialized in movies"

history = [
    {"role": "user", "parts": system_prompt}
]

print("Assistant: Hi, type 'exit' to quit the chat")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye")
        break

    history.append({"role":"user", "parts": user_input})

    start_time = time.time()

    response = model.generate_content(
        history,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1500
        }
    )

    end_time = time.time()
    duration = round(end_time-start_time)

    try:
        output = response.text
    except Exception:
        output = "No valid response text returned"

    print(f"Assistant: \n {response.text}\n duration: {duration}")

    # history.append({"role": "model", "parts": output})