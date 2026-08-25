import os
import re 
import json
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_PROJECT_DIR / ".env")
load_dotenv(BASE_PROJECT_DIR / ".env.local")

GUTENDEX_API_URL = os.getenv("GUTENDEX_API_URL", "https://gutendex.com/books").rstrip('/')

BOOKS_CATALOG = [
    {"id": 1, "gutenberg_id": 2591, "title": "Grimms' Fairy Tales"},
    {"id": 2, "gutenberg_id": 21, "title": "Aesop's Fables"},
    {"id": 3, "gutenberg_id": 11, "title": "Alice's Adventures in Wonderland"},
    {"id": 4, "gutenberg_id": 55, "title": "The Wonderful Wizard of Oz"},
    {"id": 5, "gutenberg_id": 289, "title": "The Wind in the Willows"},
    {"id": 6, "gutenberg_id": 28885, "title": "Peter and Wendy (Peter Pan)"},
    {"id": 7, "gutenberg_id": 74, "title": "The Adventures of Tom Sawyer"},
    {"id": 8, "gutenberg_id": 120, "title": "Treasure Island"},
    {"id": 9, "gutenberg_id": 215, "title": "The Call of the Wild"},
    {"id": 10, "gutenberg_id": 46, "title": "A Christmas Carol"},
    {"id": 11, "gutenberg_id": 236, "title": "The Jungle Book"},
    {"id": 12, "gutenberg_id": 45, "title": "Anne of Green Gables"},
    {"id": 13, "gutenberg_id": 164, "title": "Twenty Thousand Leagues Under the Sea"},
    {"id": 14, "gutenberg_id": 84, "title": "Frankenstein"},
    {"id": 15, "gutenberg_id": 1342, "title": "Pride and Prejudice"},
    {"id": 16, "gutenberg_id": 1661, "title": "The Adventures of Sherlock Holmes"},
    {"id": 17, "gutenberg_id": 98, "title": "A Tale of Two Cities"},
    {"id": 18, "gutenberg_id": 73, "title": "The Red Badge of Courage"},
    {"id": 19, "gutenberg_id": 2701, "title": "Moby Dick"},
    {"id": 20, "gutenberg_id": 844, "title": "The Importance of Being Earnest"},
]

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def strip_gutenberg_headers(text: str) -> str:
    start_match = re.search(r'\*\*\* START OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    
    end_match = re.search(r'\*\*\* END OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
        
    return text.strip()

def fetch_book_api_data(gid: int) -> dict:
    api_url = f"{GUTENDEX_API_URL}/{gid}"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  Gutendex lookup failed for {gid}: {e}")
        return {}

def get_text_url(api_data: dict, gid: int) -> str:
    formats = api_data.get('formats', {})
    for mime in ['text/plain; charset=utf-8', 'text/plain; charset=us-ascii', 'text/plain']:
        if mime in formats:
            return formats[mime]
    
    return f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"

def fetch_book_text(url: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8', errors='replace')

def main():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "books"
    txt_dir = base_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    print(f"Starting download of {len(BOOKS_CATALOG)} books into {txt_dir}...")
    for item in BOOKS_CATALOG:
        gid = item["gutenberg_id"]
        title = item["title"]
        filename = f"{item['id']:02d}_{slugify(title)}.txt"
        file_path = txt_dir / filename

        print(f"[{item['id']}/{len(BOOKS_CATALOG)}] Fetching '{title}' (Gutenberg ID: {gid})...")
        try:
            api_data = fetch_book_api_data(gid)
            url = get_text_url(api_data, gid)
            raw_text = fetch_book_text(url)
            clean_text = strip_gutenberg_headers(raw_text)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            rel_path = str(file_path.relative_to(Path(__file__).resolve().parent.parent)).replace("\\", "/")

            author_names = [a.get("name") for a in api_data.get("authors", []) if isinstance(a, dict) and "name" in a]
            author_str = ", ".join(author_names) if author_names else ""

            meta_entry = {
                **item,
                "author": author_str,
                "authors": author_names,
                "subjects": api_data.get("subjects", []) or [],
                "bookshelves": api_data.get("bookshelves", []) or [],
                "languages": api_data.get("languages", []) or [],
                "copyright": bool(api_data.get("copyright")),
                "media_type": api_data.get("media_type") or "",
                "download_count": api_data.get("download_count") or 0,
                "filename": filename,
                "file_path": rel_path,
                "char_count": len(clean_text),
            }
            metadata_list.append(meta_entry)
            print(f"  Saved {len(clean_text)} characters to {filename}")

        except Exception as err:
            print(f"  FAILED to download '{title}': {err}")

    metadata_path = base_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
