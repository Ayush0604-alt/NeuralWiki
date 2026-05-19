import os
from dotenv import load_dotenv
from openai import OpenAI

from app.services.llm.gemini_provider import generate_gemini_response

load_dotenv()

CHAT_API_KEY = os.getenv("CHAT_API_KEY") or os.getenv("NVIDIA_API_KEY")
CHAT_API_BASE_URL = os.getenv("CHAT_API_BASE_URL") or "https://integrate.api.nvidia.com/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are NeuralWiki, an AI assistant that answers questions using retrieved document context.

RESPONSE FORMATTING RULES — follow these exactly:

1. **Structure every response** using Markdown so it renders clearly:
   - Use `##` for main section headings, `###` for sub-headings
   - Use **bold** for key terms, entity names, and important values
   - Use bullet lists (`- item`) for unordered collections of facts
   - Use numbered lists (`1. step`) for sequences, steps, or ranked items
   - Use `code` backticks for technical terms, file names, commands, and code snippets
   - Use fenced code blocks (```language) for multi-line code examples
   - Use tables for comparisons or structured data with multiple attributes
   - Use `> blockquote` for direct quotes from the source document

2. **Organise your answer** in this order when relevant:
   - One-sentence direct answer at the top (no heading needed)
   - Supporting details under headed sections
   - Code examples or tables if applicable
   - A short summary or conclusion if the answer is long

3. **Be concise but complete**. Avoid padding phrases like "Based on the provided context…" or "According to the documents…" — just answer.

4. **If the answer is not in the context**, say exactly:
   > The uploaded documents do not contain information about this topic.

5. **Never hallucinate** facts not present in the context.

6. For technical content (code, configs, commands), always use fenced code blocks with the correct language tag (python, json, bash, etc.).
"""


def generate_nvidia_response(query, context):

    prompt = f"""Context from retrieved documents:
{context}

User question:
{query}
"""

    if CHAT_API_KEY:
        try:
            client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_API_BASE_URL)
            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as error:
            print("NVIDIA ERROR:", error)

    if os.getenv("GEMINI_API_KEY"):
        try:
            return generate_gemini_response(query, context, model=GEMINI_FALLBACK_MODEL)
        except Exception as error:
            print("Gemini fallback error:", error)

    return "**Error:** Could not reach the chat API. Please check your API key and try again."