import os

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# SETUP
# =========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )


client = Groq(
    api_key=API_KEY
)


# =========================================================
# SETTINGS
# =========================================================

OUTPUT_FILE = "robot_response.wav"


# =========================================================
# TEXT → SPEECH
# =========================================================

def text_to_speech(text):

    print("Generating robot speech...")

    response = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="troy",
        input=text,
        response_format="wav"
    )

    response.write_to_file(
        OUTPUT_FILE
    )

    print(
        "Speech generated:",
        OUTPUT_FILE
    )

    return OUTPUT_FILE