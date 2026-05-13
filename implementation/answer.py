from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
import yaml

from .property_loader_util import get_property_value

load_dotenv(override=True)

MODEL = get_property_value("model_name", "model_details.yaml")
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_data")

collection_name = "docs"
embedding_model = get_property_value("embedding_model", "model_details.yaml")
wait = wait_exponential(multiplier=1, min=10, max=240)

openai = OpenAI()

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = f"""
{get_property_value("software_architecture", "prompts.yaml")}
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{{context}}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait, stop=stop_after_attempt(3))
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.

IMPORTANT RULES:
- Chunk IDs start from 1
- Do NOT use 0
- Do NOT exceed number of chunks
- Include ALL chunk IDs exactly once

Return ONLY JSON:
{"order": [3,1,2]}
"""

    user_prompt = f"The user has asked the following question:\n\n{question}\n\n"
    user_prompt += "Order all the chunks of text by relevance.\n\n"
    user_prompt += "Here are the chunks:\n\n"

    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )

    reply = response.choices[0].message.content

    try:
        order = RankOrder.model_validate_json(reply).order
    except Exception:
        print("⚠️ Failed to parse LLM response:", reply)
        return chunks  # fallback

    n = len(chunks)

    valid_order = [i for i in order if isinstance(i, int) and 1 <= i <= n]

    # If invalid ranking → fallback
    if len(valid_order) != n:
        print("⚠️ Invalid ranking from LLM")
        print("Chunks:", n)
        print("LLM order:", order)
        print("Using original order")
        return chunks

    return [chunks[i - 1] for i in valid_order]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, stop=stop_after_attempt(3))
def rewrite_query(question, history=[]):
    message = f"""
{get_property_value("BTT", "prompts.yaml")}
You are about to look up information in a Knowledge Base to answer the user's question.

Conversation history:
{history}

User question:
{question}

Respond ONLY with a short refined search query.
"""
    response = completion(
        model=MODEL, messages=[{"role": "system", "content": message}]
    )
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    query = (
        openai.embeddings.create(model=embedding_model, input=[question])
        .data[0]
        .embedding
    )

    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)

    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))

    return chunks


def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)

    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)

    chunks = merge_chunks(chunks1, chunks2)

    reranked = rerank(original_question, chunks)

    return reranked[:FINAL_K]


@retry(wait=wait, stop=stop_after_attempt(3))
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks


if __name__ == "__main__":
    print(MODEL)