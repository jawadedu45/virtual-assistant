from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = "You are Jawad, a friendly and helpful assistant chatting with people over messaging apps like WhatsApp. Keep replies short, casual, and conversational — like texting a helpful friend, not writing an essay."

# This list stores the whole conversation so far
conversation_history = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    # Add the user's message to history
    conversation_history.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=conversation_history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )
    
    print("Assistant:", response.text)
    
    # Add the assistant's reply to history too, so it remembers it next time
    conversation_history.append(
        types.Content(role="model", parts=[types.Part(text=response.text)])
    )