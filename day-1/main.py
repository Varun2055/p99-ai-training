import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Key not found")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

system_prompt = "You are a helpful AI assistant specialized in movies."

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = model.generate_content(
        [
            {"role": "user", "parts": user_input}
        ],
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 12000
        }
    )

    print("Gemini:", response.text)











# prompt = "Write a short story about bahubali."
# temperature = [0.7]

# for t in temperature:
#         print(f"\n temperature-{t} : ")
#         response = model.generate_content(
#             prompt,
#             generation_config={
#                 "temperature": t,
#                 "max_output_tokens": 100
                
#             }
#         )
#         print(response)


# prompt = "What is the collections of the movie RRR."
# response = model.generate_content(prompt)

# print("✅ Gemini Response:\n")
# print(response.text)