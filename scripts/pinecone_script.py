import json
import os
import sys
import glob
import time
from pathlib import Path
from dotenv import load_dotenv
import httpx
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Environment & Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "https://agent-teacher-index-zugzqii.svc.aped-4627-b74a.pinecone.io")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_namespace") 

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY", "")
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai")
LLMOD_EMBEDDING_MODEL = os.getenv("LLMOD_EMBEDDING_MODEL", "MB5R2CF-azure/text-embedding-3-small")
EMBEDDINGS_URL = f"{LLMOD_BASE_URL}/v1/embeddings"

BOOKS_DIR = BASE_DIR / "data" / "books"
TEXT_FILES_PATH = str(BOOKS_DIR / "txt" / "*.txt")
JSON_METADATA_PATH = str(BOOKS_DIR / "metadata.json")

# Chunking Hyperparameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))

if not PINECONE_API_KEY or PINECONE_API_KEY in ["your_pinecone_api_key_here", "API_KEY", ""]:
    print("\n[ERROR] PINECONE_API_KEY is missing or unset in your .env file!")
    print("Please open the '.env' file in your project root and set:")
    print("PINECONE_API_KEY=pcsk_your_actual_key_here\n")
    sys.exit(1)

if not LLMOD_API_KEY or LLMOD_API_KEY in ["your_llmod_api_key_here", "API_KEY", ""]:
    print("\n[ERROR] LLMOD_API_KEY is missing or unset in your .env file!")
    print("Please open the '.env' file in your project root and set:")
    print("LLMOD_API_KEY=your_actual_key_here\n")
    sys.exit(1)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

# Detect Index Dimension (default to 1024 if matching existing Pinecone index)
try:
    stats = index.describe_index_stats()
    index_dim = stats.get("dimension") if isinstance(stats, dict) else getattr(stats, "dimension", None)
except Exception:
    index_dim = None

EMBEDDING_DIMENSIONS = int(os.getenv("LLMOD_EMBEDDING_DIMENSIONS", str(index_dim or 1024)))

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

def get_llmod_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLMOD_API_KEY}",
    }
    payload = {"model": LLMOD_EMBEDDING_MODEL, "input": texts}
    if EMBEDDING_DIMENSIONS:
        payload["dimensions"] = EMBEDDING_DIMENSIONS

    response = httpx.post(
        EMBEDDINGS_URL,
        headers=headers,
        json=payload,
        timeout=60.0,
    )
    if not response.is_success:
        raise RuntimeError(f"LLMod embeddings request failed ({response.status_code}): {response.text}")
    payload_json = response.json()
    data = payload_json.get("data", [])
    data_sorted = sorted(data, key=lambda x: int(x.get("index", 0)))
    return [item["embedding"] for item in data_sorted]

# 2. Load Metadata
with open(JSON_METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)

# Key metadata by book integer string ID
metadata_dict = {str(item["id"]): item for item in metadata_list}

# 3. Process Text Files and Build Records
records_to_upsert = []

for filepath in glob.glob(TEXT_FILES_PATH):
    filename = os.path.basename(filepath)
    
    # Safely parse numeric book ID from filename (e.g. "01_grimms..." -> "1")
    file_id_str = str(int(filename.split('_')[0]))
    
    if file_id_str not in metadata_dict:
        print(f"Warning: No metadata found for file {filename}. Skipping.")
        continue

    file_metadata = metadata_dict[file_id_str]
    
    # Filter metadata for Pinecone compatibility (primitives & list of str only)
    clean_metadata = {}
    for k, v in file_metadata.items():
        if v is None:
            continue
        elif k == "authors" and isinstance(v, list):
            clean_metadata["authors"] = [a["name"] if isinstance(a, dict) and "name" in a else str(a) for a in v]
        elif isinstance(v, (str, int, float, bool, list)):
            clean_metadata[k] = v

    with open(filepath, 'r', encoding='utf-8') as f:
        file_content = f.read()

    chunks = text_splitter.split_text(file_content)

    for i, chunk in enumerate(chunks):
        record_id = f"{file_id_str}_chunk_{i}"
        
        record = {
            "_id": record_id,
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks),
            **clean_metadata,
        }
        
        records_to_upsert.append(record)

print(f"Prepared {len(records_to_upsert)} chunks for upsert.")

# 4. Generate LLMod Embeddings & Upsert Vectors to Pinecone
MAX_RETRIES = 4
for i in range(0, len(records_to_upsert), BATCH_SIZE):
    batch = records_to_upsert[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            texts = [r["text"] for r in batch]
            embeddings = get_llmod_embeddings(texts)
            vectors = []
            for record, embedding in zip(batch, embeddings):
                metadata = {k: v for k, v in record.items() if k != "_id"}
                vectors.append({
                    "id": record["_id"],
                    "values": embedding,
                    "metadata": metadata,
                })
            index.upsert(vectors=vectors, namespace=NAMESPACE)
            print(f"Upserted batch {batch_num}/{(len(records_to_upsert) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} records)")
            break
        except Exception as e:
            error_msg = getattr(e, 'body', str(e))
            if attempt < MAX_RETRIES:
                wait_time = attempt * 10
                print(f"Batch {batch_num} rate-limit pause (attempt {attempt}/{MAX_RETRIES}): {error_msg}. Waiting {wait_time}s...")
                time.sleep(wait_time)
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(host=INDEX_HOST)
            else:
                print(f"Error upserting batch {batch_num} after {MAX_RETRIES} attempts: {error_msg}")

    # Small delay between batches to stay under rate limits
    time.sleep(0.5)

print("Upload complete!")