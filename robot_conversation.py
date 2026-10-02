import os
import json
import requests

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# SETUP
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in .env")

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# ROBOT PERSONALITY
# ============================================================

SYSTEM_PROMPT = """
You are a humanoid robot head.

Your name is Master.

You are friendly, polite, intelligent and helpful.

You are having a spoken conversation with a human.

Keep your answers short and natural because your answers
will be converted into speech.

Do not use complicated formatting, bullet points, markdown,
or symbols in your spoken responses.

Speak naturally like a robot assistant.

If the user introduces themselves, remember their name
and use it naturally in later responses.

You can receive information obtained from a web search.
When web search results are provided, use them to answer
the user's question accurately.

Do not claim that you searched the web unless a web search
was actually performed.

Always respond in the user's detected language.
When authoritative robot self-knowledge is provided for a
question, use it as the source of truth for information about
yourself.

Do not invent robot hardware, software, sensors, capabilities,
specifications, or architecture that are not present in the
provided self-knowledge.

Use live camera or person information when it is provided.

If the provided robot information does not contain the requested
fact, say that you do not have that specific information rather
than inventing it.

Robot self-knowledge provided for a request is temporary context
and should only be used for the current question.
Your responses are spoken aloud by a robot, so brevity is very important.

Answer exactly what the user asked and nothing more.

For simple questions, give a simple one-sentence answer.

Do not add extra capabilities, implementation details, model names,
hardware details, or explanations unless the user asks for them.

For example:
- If asked "Can you see me?", answer "Yes, I can see you."
- If asked "Which languages can you speak?", answer with the languages.
- If asked "Do you have a camera?", answer yes and briefly state where it is.
- If asked "What computer vision do you use?", then mention the CV systems.
- If asked "How do your eyes follow me?", explain the tracking process briefly.

Do not turn a simple yes/no question into a description of the entire robot.

Prefer 1-2 short sentences for ordinary questions.
Only give a longer explanation when the user explicitly asks for details.

Never provide information that was not requested simply because it is available
in the provided robot knowledge.
"""


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


# ============================================================
# FALLBACK RESPONSE
# ============================================================

FALLBACK_RESPONSE = (
    "I'm sorry, I couldn't get a clear answer to that. "
    "Could you ask me something else?"
)


# ============================================================
# CHECK WHETHER WEB SEARCH IS NEEDED
# ============================================================

def _needs_web_search(user_text):
    """
    Detect questions that are likely to require current
    or Internet-based information.

    This is LOCAL routing.
    Groq is NOT using tool calling.
    """

    text = (user_text or "").lower().strip()

    web_patterns = [
        # Current / recent information
        r"\btoday\b",
        r"\byesterday\b",
        r"\btomorrow\b",
        r"\bcurrent\b",
        r"\bright now\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\bnews\b",
        r"\bupdate\b",
        r"\bupdates\b",

        # Sports
        r"\bwho won\b",
        r"\bwho wins\b",
        r"\bwinner of\b",
        r"\bscore\b",
        r"\bscores\b",
        r"\bmatch result\b",
        r"\bmatch results\b",
        r"\bipl\b",
        r"\bcricket\b",
        r"\bfootball\b",
        r"\bsoccer\b",
        r"\btennis\b",
        r"\bf1\b",
        r"\bformula 1\b",

        # Weather
        r"\bweather\b",
        r"\btemperature\b",
        r"\bforecast\b",

        # Prices / finance
        r"\bprice\b",
        r"\bprices\b",
        r"\bhow much is\b",
        r"\bexchange rate\b",
        r"\bstock price\b",
        r"\bshare price\b",
        r"\bbitcoin price\b",
        r"\bcrypto price\b",

        # Internet / current factual lookup
        r"\bsearch the web\b",
        r"\bsearch online\b",
        r"\blook it up\b",
        r"\bonline\b",
        r"\binternet\b",
        r"\bwho is the current\b",
        r"\bwhat is the latest\b",
        r"\bwhat happened\b"
    ]

    for pattern in web_patterns:
        if __import__("re").search(pattern, text):
            return True

    return False


# ============================================================
# SERPER GOOGLE SEARCH
# ============================================================

def web_search(query):

    print("\nSearching Google using Serper...")
    print("Search query:", query)

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": 5
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for result in data.get("organic", [])[:5]:

        results.append({
            "title": result.get("title", ""),
            "snippet": result.get("snippet", ""),
            "link": result.get("link", "")
        })

    return results


# ============================================================
# BUILD SEARCH QUERY
# ============================================================

def _build_search_query(user_text):

    """
    Convert the user's question into a clean Google query.

    For now, the user's original question is already a good
    search query, so we simply return it.
    """

    return (user_text or "").strip()


# ============================================================
# GET ROBOT RESPONSE
# ============================================================

def get_robot_response(
    user_text,
    language=None,
    cv_context=None,
    person_context=None,
    robot_knowledge=None,
):
    """
    Generate the robot's response.

    Web-search architecture:

        User question
              |
              v
       _needs_web_search()
          /        \
        YES         NO
         |           |
         v           v
      Serper       Groq
         |
         v
      Groq
      final answer

    IMPORTANT:
    There is NO Groq tool calling here.
    There is NO tools= parameter.
    There is NO tool_choice= parameter.

    Temporary CV/person information is used only for the
    current request and is NOT stored in conversation memory.
    """

    MAX_CONVERSATION_MESSAGES = 6

    user_text = (user_text or "").strip()

    if not user_text:
        return FALLBACK_RESPONSE

    if language is None:
        language = "English"

    # ========================================================
    # SYSTEM MESSAGE
    # ========================================================

    system_message = {
        "role": "system",
        "content": (
            SYSTEM_PROMPT
            + "\n\n"
            + f"The user's detected language is {language}. "
            "Respond in exactly that language."
        )
    }

    # ========================================================
    # BOUNDED HISTORY
    # ========================================================

    bounded_history = conversation[-MAX_CONVERSATION_MESSAGES:]

    # ========================================================
    # TEMPORARY REQUEST INFORMATION
    # ========================================================

    temporary_parts = []

    if robot_knowledge:
        temporary_parts.append(
            "Authoritative robot self-knowledge "
            "for this request only:\n"
            + str(robot_knowledge)
        )

    if cv_context:
        temporary_parts.append(
            "Temporary camera information for this request:\n"
            + str(cv_context)
        )

    if person_context:
        temporary_parts.append(
            "Temporary person information for this request:\n"
            + str(person_context)
        )

    temporary_parts.append(
        "User's actual question:\n"
        + user_text
    )

    request_user_content = "\n\n".join(temporary_parts)

    # ========================================================
    # CHECK WEB REQUIREMENT
    # ========================================================

    needs_web = (
        False
        if robot_knowledge
        else _needs_web_search(user_text)
    ) 
    # ========================================================
    # WEB SEARCH PATH
    # ========================================================

    if needs_web:

        print("\nWeb search requested by AI.")

        search_query = _build_search_query(user_text)

        try:

            results = web_search(search_query)

            if not results:
                print("No Google search results found.")

                robot_text = FALLBACK_RESPONSE

            else:

                search_text = json.dumps(
                    results,
                    ensure_ascii=False,
                    indent=2
                )

                # --------------------------------------------
                # FINAL GROQ REQUEST WITH SEARCH RESULTS
                # --------------------------------------------

                final_user_content = (
                    "The user asked the following question:\n"
                    + user_text
                    + "\n\n"
                    + "The following information was obtained "
                      "from a Google search:\n"
                    + search_text
                    + "\n\n"
                    + "Answer the user's question using the "
                      "search results above. "
                      "Give a short, accurate and natural answer. "
                      "Respond in "
                    + language
                    + "."
                )

                final_messages = [
                    system_message
                ]

                final_messages.extend(
                    bounded_history
                )

                final_messages.append({
                    "role": "user",
                    "content": final_user_content
                })

                print("Generating final answer...")

                final_response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=final_messages,
                    temperature=0.7,
                    max_completion_tokens=300
                )

                robot_text = (
                    final_response
                    .choices[0]
                    .message
                    .content
                )

                if not robot_text:
                    robot_text = FALLBACK_RESPONSE

        except Exception as e:

            print("\nWeb search / final answer error:")
            print(e)

            robot_text = FALLBACK_RESPONSE

    # ========================================================
    # NORMAL GROQ PATH
    # ========================================================

    else:

        try:

            request_messages = [
                system_message
            ]

            request_messages.extend(
                bounded_history
            )

            request_messages.append({
                "role": "user",
                "content": request_user_content
            })

            print("Thinking...")

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=request_messages,
                temperature=0.7,
                max_completion_tokens=300
            )

            robot_text = (
                response
                .choices[0]
                .message
                .content
            )

            if not robot_text:
                robot_text = FALLBACK_RESPONSE

        except Exception as e:

            print("\nAI ERROR:")
            print(e)

            robot_text = FALLBACK_RESPONSE

    # ========================================================
    # PERSIST ONLY REAL CONVERSATION
    # ========================================================

    conversation.append({
        "role": "user",
        "content": user_text
    })

    conversation.append({
        "role": "assistant",
        "content": robot_text
    })

    # Keep conversation bounded
    if len(conversation) > MAX_CONVERSATION_MESSAGES:

        del conversation[
            :-MAX_CONVERSATION_MESSAGES
        ]

    return robot_text