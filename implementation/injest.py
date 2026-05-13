from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import fitz
from .property_loader_util import get_property_value

load_dotenv(override=True)

MODEL = get_property_value("model_name", "model_details.yaml")

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_data")

collection_name = "docs"
embedding_model = get_property_value("embedding_model", "model_details.yaml")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 500
wait = wait_exponential(multiplier=1, min=10, max=240)

WORKERS = 2

openai = OpenAI()


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline
            + "\n\n"
            + self.summary
            + "\n\n"
            + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """Load PDF documents from knowledge base folders"""
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.pdf"):
            pdf_text = ""
            # Open PDF
            with fitz.open(file) as pdf:
                for page in pdf:
                    pdf_text += page.get_text()
            documents.append(
                {"type": doc_type, "source": file.as_posix(), "text": pdf_text}
            )
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

A chatbot will use these chunks to answer questions about the content.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(process_document, documents), total=len(documents)
        ):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in chroma.list_collections():
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]
    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


def append_embeddings(chunks):
    """
    Append embeddings for newly processed chunks to the existing vector store.
    Unlike create_embeddings(), this does NOT recreate the collection.
    """
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(collection_name)
    print("Appending embeddings to existing vector store...")
    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(
        model=embedding_model,
        input=texts
    ).data
    vectors = [e.embedding for e in emb]
    
    
    metas = [chunk.metadata for chunk in chunks]
    
    
    # Get current vector count to generate new IDs
    existing_count = collection.count()
    ids = [str(existing_count + i) for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metas
    )
    print(f"Added {len(chunks)} new chunks.")
    print(f"Total documents in vector store: {collection.count()}")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
