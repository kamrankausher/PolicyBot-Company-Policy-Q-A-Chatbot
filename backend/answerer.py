"""
Answerer module: builds a prompt from retrieved chunks and calls Google Gemini to generate an answer.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from backend.models import SourceChunk

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini model once at module level
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = """You are a helpful company policy assistant.
Answer questions ONLY based on the provided policy document excerpts below.
If the answer is not found in the excerpts, say: I could not find information about this in the uploaded policy document.
Never make up information.
Be concise and professional.

Policy Excerpts:
{context}"""


def generate_answer(
    question: str,
    source_chunks: list[SourceChunk],
    chat_history: list[dict],
) -> tuple[str, str]:
    """Send the question + retrieved context to Gemini and return (answer, confidence)."""
    context = "\n\n".join(
        f"[Page {chunk.page_number}, Chunk {chunk.chunk_index}]:\n{chunk.text}"
        for chunk in source_chunks
    )

    system_message = SYSTEM_PROMPT.format(context=context)

    # Build conversation messages: system + last 4 history messages + current question
    messages = [{"role": "user", "parts": [system_message]}]
    messages.append({"role": "model", "parts": ["Understood. I will answer only from the provided policy excerpts."]})

    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "model"
        messages.append({"role": role, "parts": [msg["content"]]})

    messages.append({"role": "user", "parts": [question]})

    try:
        response = gemini_model.generate_content(messages)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Sorry, I encountered an error generating the answer. Please try again. (Error: {str(e)})"
        return answer, "Low"

    # Determine confidence based on whether the model found relevant information
    confidence = "Low" if "could not find" in answer.lower() else "High"
    return answer, confidence
