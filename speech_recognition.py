import sounddevice as sd
import soundfile as sf

from dotenv import load_dotenv
from groq import Groq

from lingua import Language, LanguageDetectorBuilder

import os


# =========================================================
# SETUP
# =========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found."
    )


client = Groq(
    api_key=API_KEY
)


# =========================================================
# AUDIO SETTINGS
# =========================================================

SAMPLE_RATE = 16000
CHANNELS = 1

RECORD_SECONDS = 5

AUDIO_FILE = "recording.wav"


# =========================================================
# WAKE / EXIT WORDS
# =========================================================

WAKE_WORDS = [
    "master",
    "hey master",
    "hi master",
    "hello master"
]


EXIT_WORDS = [
    "exit",
    "stop",
    "bye",
    "goodbye",
    "quit"
]


# =========================================================
# LINGUA LANGUAGE DETECTOR
# =========================================================

languages = [

    Language.ENGLISH,

    Language.HINDI,

    Language.MARATHI

]


detector = (
    LanguageDetectorBuilder
    .from_languages(*languages)
    .build()
)


# =========================================================
# CHECK WAKE WORD
# =========================================================

def is_wake_word(text):

    if not text:
        return False

    text = text.lower().strip()

    for wake_word in WAKE_WORDS:

        if wake_word in text:
            return True

    return False


# =========================================================
# CHECK EXIT COMMAND
# =========================================================

def is_exit_command(text):

    if not text:
        return False

    text = text.lower().strip()

    for command in EXIT_WORDS:

        if command in text:
            return True

    return False


# =========================================================
# RECORD SPEECH
# =========================================================

def record_speech():

    print(
        "\nListening... Speak now."
    )


    audio = sd.rec(

        int(
            RECORD_SECONDS *
            SAMPLE_RATE
        ),

        samplerate=SAMPLE_RATE,

        channels=CHANNELS,

        dtype="float32"
    )


    sd.wait()


    sf.write(

        AUDIO_FILE,

        audio,

        SAMPLE_RATE
    )


    print(
        "Recording complete."
    )


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text):

    if not text or not text.strip():

        return "Unknown"


    try:

        detected = (
            detector.detect_language_of(
                text
            )
        )


        if detected == Language.ENGLISH:

            return "English"


        if detected == Language.HINDI:

            return "Hindi"


        if detected == Language.MARATHI:

            return "Marathi"


    except Exception as e:

        print(
            "Language detection error:",
            e
        )


    return "Unknown"


# =========================================================
# SPEECH → TEXT + LANGUAGE
# =========================================================

def speech_to_text():

    print(
        "Converting speech to text..."
    )


    with open(
        AUDIO_FILE,
        "rb"
    ) as file:

        transcription = (
            client.audio.transcriptions.create(

                file=(
                    AUDIO_FILE,
                    file.read()
                ),

                model="whisper-large-v3-turbo",

                response_format="json",

                temperature=0
            )
        )


    recognized_text = (
        transcription.text
    )


    # -------------------------------------------------------
    # Lingua detects language from Whisper text
    # -------------------------------------------------------

    detected_language = (
        detect_language(
            recognized_text
        )
    )


    print(
        "Recognized:",
        recognized_text
    )


    print(
        "Language:",
        detected_language
    )


    # -------------------------------------------------------
    # IMPORTANT:
    # Return BOTH
    # -------------------------------------------------------

    return (
        recognized_text,
        detected_language
    )