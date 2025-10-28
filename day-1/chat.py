import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API Key not found in .env file!")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize model
model = genai.GenerativeModel("gemini-2.0-flash")

# System role (optional — sets model behavior)
system_prompt = "You are a helpful AI assistant specialized in startups and technology."

# Store conversation history
history = [
    {"role": "user", "parts": system_prompt}
]

# Chat loop
print("🤖 Gemini Chatbot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Add user message to history
    history.append({"role": "user", "parts": user_input})

    # Generate response
    response = model.generate_content(
        history,
        generation_config={
            "temperature": 0.7,        # controls creativity
            "max_output_tokens": 150,  # limits response size
        }
    )

    # Extract response text safely
    try:
        output = response.text
    except Exception:
        output = "(No valid response text returned.)"

    print(f"Gemini: {output}\n")

    # Add model reply to history
    history.append({"role": "model", "parts": output})



# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("API_KEY")
# if not api_key:
#     raise ValueError("Key not found")

# genai.configure(api_key=api_key)

# model = genai.GenerativeModel("gemini-2.5-flash")

# while True:
#     prompt = input("Enter your prompt: ")
#     word_limit = input("Enter the maximum number of words for the response: ")
#     prompt = f"{prompt} with word limit of {word_limit} words."
#     response = model.generate_content(
#         prompt,
#         generation_config={
#             "temperature": 0.7
#         }
#     )
#     print(" Gemini Response:\n")
#     print(response.text)
#     over = input("Do you want to continue? (yes/no): ").strip().lower()
#     if over == 'no':
#         over = True
#     elif over == 'yes':
#         over = False
#     else:
#         print("Invalid input. Exiting.")
#         over = True