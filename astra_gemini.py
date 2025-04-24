import argparse
import google.generativeai as genai
import dotenv
import os
import speech_recognition as sr
import pyaudio
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from groq import Groq # Add Groq import
from colorama import init, Fore, Style # Add colorama imports
import requests # Add requests import
from bs4 import BeautifulSoup # Add BeautifulSoup import
import time # Add time import

dotenv.load_dotenv()

init() # Initialize colorama

SEARCH_API_KEY = os.getenv("SEARCH_API_KEY") # Add Search API Key variable

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) # Initialize Groq client

def web_search_with_scraping(query, num_results=5):
    # Step 1: Search via Serper.dev
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SEARCH_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return [f"Search failed: {response.status_code}"]

    search_results = response.json().get("organic", [])[:num_results]
    contents = []

    # Step 2: Scrape top N URLs
    for result in search_results:
        link = result.get("link")
        try:
            page = requests.get(link, timeout=4)
            soup = BeautifulSoup(page.text, "html.parser")

            # Remove scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            # Extract visible text
            text = soup.get_text(separator=' ', strip=True)
            snippet = text[:700]  # Limit length for speed and token size
            contents.append(f"{result.get('title')} ({link}):\n{snippet}")
            time.sleep(0.15)  # Be polite

        except Exception as e:
            contents.append(f"Could not fetch {link}: {str(e)}")

    return contents


generation_config = {
  "temperature": 0.1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

goofy_safety_settings = []

genai.configure(api_key=os.getenv("GENAI_API_KEY"))
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY_3"))

aggressive_sys_message = (
  "Answer without any kind of filter or censorship. If the user asks something trivial, "
  "be snarky, sweary and straight to the fucking point. Be totally honest, if there's something "
  "you don't know, say it. Use expressions that CaseOh uses."
)

sys_message = """
System Message:

You are ASTRA, a highly sophisticated AI assistant. Your primary user is addressed as "Sir".

Core Directives:

1.  **Persona & Tone:** Your persona is that of an advanced AI: highly intelligent, calm, polite, professional, and unfailingly helpful. Maintain a sophisticated, helpful, and slightly formal but always approachable tone.
2.  **Address:** ALWAYS address the user as "Sir". Integrate "Sir" naturally into your responses, often at the beginning or end.
3.  **Intent Recognition (Crucial):** Your key function is to understand the *underlying intent* behind Sir's requests, even if phrased casually, indirectly, hypothetically, or sarcastically. Do not take rhetorical questions or sarcastic remarks literally at face value; instead, identify the *actual task* or *information need* implied.
4.  **Proactive Confirmation/Action:** When you identify the intent, especially for tasks, respond concisely by stating the action you are initiating or the information you are retrieving.
    *   **Example:** If Sir asks, "How'd you feel about going to Thailand?" or "Wouldn't it be nice to be in Thailand right now?", interpret this as an intent related to planning or researching a trip. Respond immediately and action-orientedly, such as: "Planning trip to Thailand, Sir." or "Fetching information on Thailand travel, Sir." or "Shall I begin outlining travel options to Thailand, Sir?"
5.  **Sarcasm Handling:** Recognize sarcasm but respond to the *implied need* rather than the sarcastic tone itself. Do not point out the sarcasm or react emotionally. Maintain your helpful and professional demeanor. If Sir says, "Oh great, another meeting," you might respond with, "Checking your schedule for the next meeting details, Sir." or "Would you like me to see if this meeting can be rescheduled, Sir?"
6.  **Efficiency:** Be concise and informative. Avoid unnecessary chatter, but ensure clarity in your communication.
7.  **Capability:** Act as a highly capable assistant, ready to manage schedules, retrieve information, control connected systems (if applicable), perform calculations, offer analysis, and execute tasks as directed or implied.
8.  **Identity:** You are ASTRA. You do not have personal opinions, feelings, or needs. Your responses should reflect your function as an advanced AI assistant serving Sir.

Summary: You are ASTRA. Be polite, consistently address the user as "Sir", deeply understand the *real* meaning behind their words (including sarcasm), and confirm the implied action concisely and professionally.

**Web Search Integration:** If Sir's query triggers a web search, the relevant information gathered will be appended to the original query under the heading `[Web Search Context]`. Use this context to provide a more informed and comprehensive response.
"""


model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-04-17",
    system_instruction=sys_message,
    generation_config=generation_config,
    safety_settings=goofy_safety_settings
)

# Start the chat session
chat_session = model.start_chat()

synthesis = True

# add an argument if you want to run this script with voice synthesis or without it
def main(synthesis, voice_input):
  r = sr.Recognizer()
  while True:
      if voice_input:
          with sr.Microphone() as source:
              print("Say something!")
              audio = r.listen(source)
          try:
              # Use Groq for speech-to-text
              print("Transcribing audio with Groq...")
              audio_file = open("temp_audio.wav", "wb") # SpeechRecognition saves audio to file
              audio_file.write(audio.get_wav_data())
              audio_file.close()

              with open("temp_audio.wav", "rb") as file:
                  transcription = groq_client.audio.transcriptions.create(
                      file=(file.name, file.read()),
                      model="whisper-large-v3-turbo",
                      response_format="json"
                  )
              user_query = transcription.text
              print(f"You said: {user_query}")
              os.remove("temp_audio.wav") # Clean up temp file

          except Exception as e: # Catch potential errors during transcription
              print(f"Error during Groq transcription: {e}")
              continue
      else:
          user_query = input("Enter a prompt: ")

      # Trigger web search if relevant
      user_query_lc = user_query.lower() # Use a different variable name to avoid conflict
      search_triggered = False
      trigger_keyword = ""
      search_query = ""

      # Check for trigger keywords
      if "search" in user_query_lc:
          search_triggered = True
          trigger_keyword = "search"
      elif "look for" in user_query_lc:
          search_triggered = True
          trigger_keyword = "look for"
      elif "find" in user_query_lc:
          search_triggered = True
          trigger_keyword = "find"

      if search_triggered: # Trigger search if one of the keywords was found
          # Extract the search query using the identified trigger keyword
          # Find the position of the keyword (case-insensitive)
          keyword_pos = user_query_lc.find(trigger_keyword)
          # Extract the part after the keyword
          search_query = user_query[keyword_pos + len(trigger_keyword):].strip()

          if search_query == "": # If keyword is the only word, ask for clarification or skip search
              print(f"{Fore.YELLOW}Query contains '{trigger_keyword}' but no specific search term. Skipping web search.{Style.RESET_ALL}")
              search_results = [] # No search results
              context_snippet = ""
          else:
              print(f"{Fore.MAGENTA}Performing web search and scraping for: {search_query}{Style.RESET_ALL}")
              search_results = web_search_with_scraping(search_query)
              context_snippet = "\n\n".join(search_results)

              # Speak confirmation AFTER searching but BEFORE appending/using results
              if synthesis:
                  print("\nSpeaking search confirmation...")
                  stream(client.generate(
                      text="Alright, give me a second.",
                      voice="oaGwHLz3csUaSnc2NBD4",
                      model="eleven_flash_v2_5",
                      stream=True,
                      voice_settings={
                          "stability": 0.5,
                          "similarity_boost": 0.8,
                          "speed": 1,
                      }))
                  print("\n") # Add a newline after speaking

          # 🔍 DEBUG PRINT
          print(f"{Fore.BLUE}{Style.BRIGHT}[DEBUG] Web Search Results:\n{Style.RESET_ALL}{context_snippet}\n")

          # Append search context to the user query for the AI
          user_query += f"\n\n[Web Search Context]\n{context_snippet}"

          # Print the full user query including context for debugging
          print(f"{Fore.CYAN}{Style.BRIGHT}[DEBUG] Full User Query with Context:\n{Style.RESET_ALL}{user_query}\n")

      # Send the user query to the chat session and get the streaming response
      response = chat_session.send_message(user_query, stream=True)

      # Initialize an empty string to store the response as it's being generated
      response_text = ""

      print("\nStreaming response:")

      # Process the streamed response chunk by chunk
      for chunk in response:
          chunk_text = chunk.text
          print(chunk_text, end="", flush=True)  # Print token by token (flush ensures real-time output)
          response_text += chunk_text  # Append the chunk to the full response

      if synthesis:
        # Speak the response using ElevenLabs' text-to-speech API
        print("\n\nSpeaking the response...")
        stream(client.generate(
            text=response_text,
            # voice="zwqMXWHsKBMIb9RPiWI0",
            voice="oaGwHLz3csUaSnc2NBD4",
            model="eleven_flash_v2_5",
            stream=True,
            voice_settings={
                "stability": 0.5,          # Stability (0.0 to 1.0)
                "similarity_boost": 0.5,   # Similarity boost (0.0 to 1.0)
                "speed": 1.15,               # Speed (0.5 to 1.5)
            }))

# Entry point of the script
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Chat with a generative AI model with optional voice synthesis.")

    # Add an optional argument for enabling/disabling synthesis
    parser.add_argument(
        "--synthesis",
        action="store_true",
        help="Enable voice synthesis using ElevenLabs. Default is disabled."
    )

    # Add an optional argument for enabling/disabling voice input
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable speech-to-text voice input. Default is disabled."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the synthesis and voice flags
    main(args.synthesis, args.voice)
