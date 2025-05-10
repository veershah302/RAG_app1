import os
import sqlite3
import json
import asyncio



# pip install aiofiles
import aiofiles



import threading

import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory

# pip install flask_session.Session

from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import chromadb
import torch
# pip install fitz
# pip install frontend
# pip install tools
import fitz
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# App setup
app = Flask(__name__, static_folder="static")
app.config["SECRET_KEY"] = "your-secret-key"  # Change this in production
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CHROMA_DB_PATH = "chroma_db"
API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
ALLOWED_EXTENSIONS = {"pdf"}
SIMILARITY_THRESHOLD = 0.9
CHUNK_SIZE = 200
BATCH_SIZE = 5000

# Initialize CLIP
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Initialize BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Route to serve files from 'users' directory
@app.route('/users/<path:filename>')
def serve_users(filename):
    logger.debug(f"Serving file: {filename}")
    return send_from_directory('users', filename, as_attachment=False)

# SQLite setup
def init_db():
    logger.info("Initializing database")
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            username TEXT,
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            step TEXT,
            progress INTEGER,
            total INTEGER,
            done INTEGER,
            error TEXT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_tasks (
            username TEXT,
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            mode TEXT,
            step TEXT,
            progress INTEGER,
            total INTEGER,
            done INTEGER,
            response TEXT,
            images TEXT,
            error TEXT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized")

init_db()

# Dedicated event loop for async tasks
async_loop = asyncio.new_event_loop()
def run_loop():
    asyncio.set_event_loop(async_loop)
    async_loop.run_forever()

threading.Thread(target=run_loop, daemon=True).start()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_clip_embedding(content, is_image=False):
    logger.debug(f"Generating embedding for {'image' if is_image else 'text'}: {content[:50]}...")
    try:
        if is_image:
            image = Image.open(content).convert("RGB")
            inputs = blip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                caption_ids = blip_model.generate(**inputs)
            caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
            clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding = clip_model.get_image_features(**clip_inputs)
            text_inputs = clip_processor(text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            with torch.no_grad():
                caption_embedding = clip_model.get_text_features(**text_inputs)
            logger.debug(f"Caption generated: {caption}")
            return image_embedding.squeeze().cpu().numpy(), caption_embedding.squeeze().cpu().numpy(), caption
        else:
            inputs = clip_processor(text=[content], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            with torch.no_grad():
                embedding = clip_model.get_text_features(**inputs)
            logger.debug("Embedding generated successfully")
            return embedding.squeeze().cpu().numpy(), None, None
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None, None, None

def query_vector_db(query_text, username, mode="image+text", top_k=5):
    logger.debug(f"Querying vector DB for user: {username}, query: {query_text}, mode: {mode}")
    try:
        collection = chroma_client.get_collection(name=f"{username}_embeddings")
        embedding_tuple = get_clip_embedding(query_text)
        if embedding_tuple is None or embedding_tuple[0] is None:
            logger.error("Failed to generate query embedding")
            return [], []
        query_embedding = embedding_tuple[0]
        logger.debug(f"Query embedding type: {type(query_embedding)}, shape: {query_embedding.shape if isinstance(query_embedding, np.ndarray) else 'N/A'}")
        
        if mode == "image":
            where_clause = {"type": "image"}
        elif mode == "text":
            where_clause = {"type": "text"}
        elif mode == "image+text":
            where_clause = None
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "embeddings"]
        )
        metadatas = results["metadatas"][0]
        embeddings = results["embeddings"][0] if results["embeddings"] else []
        logger.debug(f"Query returned {len(metadatas)} results, embeddings count: {len(embeddings)}")
        for i, (meta, emb) in enumerate(zip(metadatas, embeddings)):
            logger.debug(f"Result {i}: metadata={meta}, embedding type={type(emb)}, shape={np.array(emb).shape if emb is not None else 'None'}")
        return metadatas, embeddings
    except Exception as e:
        logger.error(f"Error querying vector DB: {e}")
        return [], []

def generate_response_with_llm(query, context):
    logger.debug(f"Context provided: {context[:100]}...")
    logger.debug(f"Generating LLM response for query: {query}")
    prompt = (
        f"Context:\n{context}\n\n"
        f"Query: {query}\n"
        "Answer: Based on the provided context, respond to the query in a structured, concise, and readable manner. "
        "Focus only on information directly relevant to the query, ignoring irrelevant details. "
        "Use numbered citations (e.g., [1]) after each key point or line, referencing the page number from the context. "
        "Do not include the raw context string or full source citations in your response. "
        "At the end, include a citation key listing all references (e.g., [1] = Page X of PDF Y)."
    )
    data = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": "You are an AI assistant for deep learning textbooks..."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "stream": True
    }
    try:
        response = requests.post(API_URL, json=data, headers=HEADERS, stream=True)
        if response.status_code != 200:
            logger.error(f"LLM request failed with status: {response.status_code}")
            return ["Error contacting LLM."]
        
        tokens = []
        for chunk in response.iter_lines():
            if chunk and chunk.startswith(b"data: "):
                chunk_data = chunk[6:].decode("utf-8")
                if chunk_data == "[DONE]":
                    break
                chunk_json = json.loads(chunk_data)
                token = chunk_json["choices"][0]["delta"].get("content", "")
                if token:
                    tokens.append(token)
        logger.debug(f"LLM returned {len(tokens)} tokens")
        return tokens
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return ["Error generating response"]

def update_task_status(username, task_id, step, progress, total, done, error=None):
    logger.debug(f"Updating task status: {username}, task_id={task_id}, step={step}, progress={progress}/{total}, done={done}, error={error}")
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO tasks (username, task_id, step, progress, total, done, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, task_id, step, progress, total, int(done), error))
    conn.commit()
    conn.close()

def update_chat_task_status(username, task_id, query, mode, step, progress, total, done, response=None, images=None, error=None):
    logger.debug(f"Updating chat task status: {username}, task_id={task_id}, query={query}, mode={mode}, step={step}, progress={progress}/{total}, done={done}, response={response[:50] if response else None}, images={images}, error={error}")
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    images_json = json.dumps(images) if images else None
    c.execute("""
        INSERT OR REPLACE INTO chat_tasks (username, task_id, query, mode, step, progress, total, done, response, images, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, task_id, query, mode, step, progress, total, int(done), response, images_json, error))
    conn.commit()
    conn.close()

async def extract_text_and_images_from_pdf(pdf_path, output_folder, username, task_id):
    logger.debug(f"Starting extraction for {pdf_path}")
    update_task_status(username, task_id, "Extracting text and images", 0, 100, False)
    try:
        doc = fitz.open(pdf_path)
        text_by_page = {}
        image_paths = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text() or ""
            text_by_page[page_num + 1] = text
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(output_folder, f"{os.path.basename(pdf_path)}_p{page_num+1}_i{img_idx+1}.{image_ext}")
                async with aiofiles.open(image_path, "wb") as f:
                    await f.write(image_bytes)
                image_paths.append((image_path, page_num + 1))
            update_task_status(username, task_id, "Extracting text and images", (page_num + 1) * 50 // total_pages, 100, False)
        update_task_status(username, task_id, "Text and images extracted", 50, 100, False)
        logger.debug(f"Extraction completed for {pdf_path}: {len(text_by_page)} pages, {len(image_paths)} images")
        return text_by_page, image_paths
    except Exception as e:
        logger.error(f"Error extracting text and images from {pdf_path}: {e}")
        update_task_status(username, task_id, "Error", 50, 100, False, str(e))
        raise

def chunk_text(text, chunk_size=CHUNK_SIZE):
    logger.debug("Chunking text")
    if not text:
        return []
    chunks = []
    current_chunk = ""
    paragraphs = text.split("\n")
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    logger.debug(f"Text chunked into {len(chunks)} chunks")
    return chunks

def process_chunks_and_images(text_by_page, image_paths, pdf_path, username, task_id):
    logger.debug("Processing chunks and images")
    update_task_status(username, task_id, "Processing chunks and images", 50, 100, False)
    embeddings = []
    metadatas = []
    total_items = sum(len(chunk_text(text)) for text in text_by_page.values()) + len(image_paths) * 2
    
    processed = 0
    try:
        for page_num, text in text_by_page.items():
            chunks = chunk_text(text)
            for chunk in chunks:
                embedding, _, _ = get_clip_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding.tolist())
                    metadatas.append({"type": "text", "content": chunk, "source": os.path.basename(pdf_path), "page": page_num})
                processed += 1
                update_task_status(username, task_id, "Processing chunks and images", 50 + (processed * 50 // total_items), 100, False)
        
        for image_path, page_num in image_paths:
            image_embedding, caption_embedding, caption = get_clip_embedding(image_path, is_image=True)
            if image_embedding is not None:
                embeddings.append(image_embedding.tolist())
                metadatas.append({"type": "image", "path": image_path, "description": caption, "source": os.path.basename(pdf_path), "page": page_num})
                processed += 1
                update_task_status(username, task_id, "Processing chunks and images", 50 + (processed * 50 // total_items), 100, False)
            if caption_embedding is not None:
                embeddings.append(caption_embedding.tolist())
                metadatas.append({"type": "caption", "content": caption, "source": os.path.basename(pdf_path), "page": page_num, "image_path": image_path})
                processed += 1
                update_task_status(username, task_id, "Processing chunks and images", 50 + (processed * 50 // total_items), 100, False)
        logger.debug(f"Processed {processed} items: {len(embeddings)} embeddings generated")
        return embeddings, metadatas
    except Exception as e:
        logger.error(f"Error processing chunks and images: {e}")
        update_task_status(username, task_id, "Error", 50 + (processed * 50 // total_items), 100, False, str(e))
        raise

def add_to_chroma_in_batches(collection, embeddings, metadatas, ids, username, task_id):
    logger.debug("Storing embeddings in ChromaDB")
    update_task_status(username, task_id, "Storing embeddings in database", 0, len(embeddings), False)
    try:
        for i in range(0, len(embeddings), BATCH_SIZE):
            batch_embeddings = embeddings[i:i + BATCH_SIZE]
            batch_metadatas = metadatas[i:i + BATCH_SIZE]
            batch_ids = ids[i:i + BATCH_SIZE]
            collection.add(embeddings=batch_embeddings, metadatas=batch_metadatas, ids=batch_ids)
            update_task_status(username, task_id, "Storing embeddings in database", min(i + BATCH_SIZE, len(embeddings)), len(embeddings), False)
            logger.debug(f"Stored batch {i // BATCH_SIZE + 1}: {len(batch_embeddings)} items")
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        update_task_status(username, task_id, "Error", min(i + BATCH_SIZE, len(embeddings)), len(embeddings), False, str(e))
        raise

async def process_pdf_folder(pdf_folder, username, task_id):
    logger.info(f"Starting preprocessing for user: {username}, folder: {pdf_folder}, task_id: {task_id}")
    update_task_status(username, task_id, "Initializing", 0, 100, False)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        update_task_status(username, task_id, "Completed", 100, 100, True)
        logger.info("No PDFs found to process")
        return
    
    try:
        collection = chroma_client.get_collection(name=f"{username}_embeddings")
        chroma_client.delete_collection(name=f"{username}_embeddings")
    except:
        pass
    try:
        collection = chroma_client.create_collection(name=f"{username}_embeddings", metadata={"dimension": 512})
        embeddings = []
        metadatas = []
        next_id = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text_by_page, image_paths = await extract_text_and_images_from_pdf(pdf_path, pdf_folder, username, task_id)
            pdf_embeddings, pdf_metadatas = process_chunks_and_images(text_by_page, image_paths, pdf_path, username, task_id)
            embeddings.extend(pdf_embeddings)
            metadatas.extend(pdf_metadatas)
        
        if embeddings:
            ids = [str(i) for i in range(next_id, next_id + len(embeddings))]
            add_to_chroma_in_batches(collection, embeddings, metadatas, ids, username, task_id)
        update_task_status(username, task_id, "Completed", 100, 100, True)
        logger.info(f"Preprocessing completed for user: {username}, task_id: {task_id}")
    except Exception as e:
        update_task_status(username, task_id, "Error", 50, 100, False, str(e))
        logger.error(f"Preprocessing failed for user {username}, task_id {task_id}: {e}")

async def process_chat_query(username, task_id, query, mode):
    try:
        # Step 1: Generate query embedding
        logger.info(f"Starting chat processing for user: {username}, task_id: {task_id}, query: {query}, mode: {mode}")
        update_chat_task_status(username, task_id, query, mode, "Creating embedding", 0, 100, False)
        query_embedding = generate_query_embedding(query)
        if query_embedding is None:
            raise Exception("Failed to generate query embedding")
        update_chat_task_status(username, task_id, query, mode, "Matching", 25, 100, False)

        # Step 2: Handle queries and matching based on mode
        if mode == "image+text":
            # Query image, text, and captions independently
            image_metadatas, image_embeddings = query_vector_db(query, username, "image")
            text_metadatas, text_embeddings = query_vector_db(query, username, "text")
            caption_metadatas, caption_embeddings = query_vector_db(query, username, "image+text")

            # Validate each set independently
            valid_images = validate_results(image_metadatas, image_embeddings)
            valid_texts = validate_results(text_metadatas, text_embeddings)
            valid_captions = validate_results(caption_metadatas, caption_embeddings)

            if not valid_images:
                logger.warning(f"No valid image results: metadatas count={len(image_metadatas)}, embeddings count={len(image_embeddings)}")
                image_metadatas, image_embeddings = [], []
            if not valid_texts:
                logger.warning(f"No valid text results: metadatas count={len(text_metadatas)}, embeddings count={len(text_embeddings)}")
                text_metadatas, text_embeddings = [], []
            if not valid_captions:
                logger.warning(f"No valid caption results: metadatas count={len(caption_metadatas)}, embeddings count={len(caption_embeddings)}")
                caption_metadatas, caption_embeddings = [], []

            # Get top 3 image and caption matches
            image_caption_response, image_paths = check_for_matches(query_embedding, 
                                                                  image_metadatas + caption_metadatas, 
                                                                  image_embeddings + caption_embeddings, 
                                                                  username, 
                                                                  "image+text")
            
            # Build text context and summarize with LLM
            context, _ = build_context(text_metadatas, username)
            text_response = ""
            if context.strip():
                update_chat_task_status(username, task_id, query, mode, "LLM generating", 75, 100, False)
                text_response = generate_response_with_llm(query, context)
            
            # Combine responses
            final_response = ""
            if image_caption_response:
                final_response += image_caption_response
            if text_response:
                if final_response:
                    final_response += "\n\nRelevant Text Summary:\n" + text_response
                else:
                    final_response = "Relevant Text Summary:\n" + text_response
            if not final_response:
                final_response = "No relevant image, caption, or text data found."

            update_chat_task_status(username, task_id, query, mode, "Completed", 100, 100, True, 
                                    response=final_response, images=image_paths or [])
            return

        else:
            # Single mode processing
            metadatas, embeddings = query_vector_db(query, username, mode)
            if not validate_results(metadatas, embeddings):
                logger.warning(f"Invalid results for {mode} mode: metadatas count={len(metadatas)}, embeddings count={len(embeddings)}")
                metadatas, embeddings = [], []

            if mode == "image":
                response, image_paths = check_for_matches(query_embedding, metadatas, embeddings, username, mode)
                if response:
                    update_chat_task_status(username, task_id, query, mode, "Completed", 100, 100, True, 
                                            response=response, images=image_paths)
                else:
                    update_chat_task_status(username, task_id, query, mode, "Completed", 100, 100, True, 
                                            response="No image matches found.", images=[])
                return

            # For "text" mode
            context, image_paths = build_context(metadatas, username)
            if context.strip():
                update_chat_task_status(username, task_id, query, mode, "LLM generating", 75, 100, False)
                response = generate_response_with_llm(query, context)
            else:
                response = "No relevant content available."
            update_chat_task_status(username, task_id, query, mode, "Completed", 100, 100, True, 
                                    response=response, images=image_paths)

    except Exception as e:
        logger.error(f"Chat processing failed for user {username}, task_id {task_id}: {e}")
        update_chat_task_status(username, task_id, query, mode, "Error", 50, 100, True, error=str(e))

        
def generate_query_embedding(query):
    try:
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return None
    

def query_vector_db(query, username, mode, top_k=5):
    try:
        collection = chroma_client.get_collection(name=f"{username}_embeddings")
        query_embedding = generate_query_embedding(query)
        if query_embedding is None:
            return [], []
        
        if mode == "image":
            where_clause = {"type": "image"}
        elif mode == "text":
            where_clause = {"type": "text"}
        else:  # "image+text" retrieves all types
            where_clause = None
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "embeddings"]
        )
        metadatas = results["metadatas"][0]
        embeddings = results["embeddings"][0] if results["embeddings"] else []
        return metadatas, embeddings
    except Exception as e:
        logger.error(f"Error querying vector DB: {e}")
        return [], []
    
def combine_mode_results(query, username):
    image_metadatas, image_embeddings = query_vector_db(query, username, mode="image")
    text_metadatas, text_embeddings = query_vector_db(query, username, mode="text")
    all_metadatas, all_embeddings = query_vector_db(query, username, mode="image+text")
    caption_metadatas = [m for m in all_metadatas if m.get("type") == "caption"]
    caption_embeddings = [e for m, e in zip(all_metadatas, all_embeddings) if m.get("type") == "caption"]
    
    metadatas = image_metadatas + text_metadatas + caption_metadatas
    embeddings = image_embeddings + text_embeddings + caption_embeddings
    return metadatas, embeddings
def check_for_matches(query_embedding, metadatas, embeddings, username, mode):
    logger.debug(f"Checking for matches in mode: {mode}")
    if mode not in ["image", "image+text"]:
        return None, None
    
    query_embedding = query_embedding.reshape(1, -1)
    
    # Separate lists for image and caption results
    image_results = []
    caption_results = []

    for metadata, embedding in zip(metadatas, embeddings):
        if embedding is None:
            logger.debug("Skipping None embedding")
            continue
        
        try:
            embedding = np.array(embedding).reshape(1, -1)
            if embedding.shape[1] != query_embedding.shape[1]:
                logger.error(f"Embedding shape mismatch: query {query_embedding.shape}, embedding {embedding.shape}")
                continue
            similarity = cosine_similarity(query_embedding, embedding)[0][0]
            logger.debug(f"Similarity: {similarity} for {metadata.get('description', metadata.get('content', 'unknown'))}")

            # Categorize based on type
            if metadata.get("type") == "image":
                image_results.append((similarity, metadata, embedding))
            elif metadata.get("type") == "caption":
                caption_results.append((similarity, metadata, embedding))
        
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
            continue
    
    # Sort by similarity (descending) and pick top 3
    image_results.sort(reverse=True, key=lambda x: x[0])
    caption_results.sort(reverse=True, key=lambda x: x[0])
    
    # Prepare responses and image paths
    responses = []
    image_paths = []

    # Top 3 image matches
    for i, (similarity, metadata, _) in enumerate(image_results[:3], 1):
        response = f"Top {i} image match (similarity: {similarity:.4f}) on page {metadata.get('page')} of {metadata.get('source')}"
        full_image_path = os.path.join("users", username, os.path.basename(metadata.get("path", "")))
        responses.append(response)
        image_paths.append(full_image_path)
        logger.info(f"{response}, path={full_image_path}")

    # Top 3 caption matches
    for i, (similarity, metadata, _) in enumerate(caption_results[:3], 1):
        response = f"Top {i} caption match (similarity: {similarity:.4f}): '{metadata.get('content')}' on page {metadata.get('page')} of {metadata.get('source')}"
        full_image_path = os.path.join("users", username, os.path.basename(metadata.get("image_path", "")))
        responses.append(response)
        image_paths.append(full_image_path)
        logger.info(f"{response}, path={full_image_path}")

    if responses:
        return "\n".join(responses), image_paths
    return None, None
def validate_results(metadatas, embeddings):
    logger.debug(f"Validating results: metadatas count={len(metadatas)}, embeddings count={len(embeddings)}")
    return len(metadatas) > 0 and len(embeddings) > 0 and len(metadatas) == len(embeddings)
def build_context(metadatas, username):
    context = ""
    image_paths = []
    for metadata in metadatas:
        if metadata.get("type") == "text":
            context += f"{metadata.get('content', '')}\n[Page: {metadata.get('page', '')} of {metadata.get('source', '')}]\n"
        elif metadata.get("type") == "image":
            context += f"[Image: {metadata.get('description', '')}\n[Page: {metadata.get('page', '')} of {metadata.get('source', '')}]\n"
            full_image_path = os.path.join("users", username, os.path.basename(metadata.get("path", "")))
            image_paths.append(full_image_path)
        elif metadata.get("type") == "caption":
            context += f"[Caption: {metadata.get('content', '')}\n[Page: {metadata.get('page', '')} of {metadata.get('source', '')}]\n"
            full_image_path = os.path.join("users", username, os.path.basename(metadata.get("image_path", "")))
            image_paths.append(full_image_path)
    return context, image_paths

def generate_response_with_llm(query, context):
    prompt = (
        f"Context:\n{context}\n\n"
        f"Query: {query}\n"
        "Answer: Based on the provided context, respond to the query in a structured, concise, and readable manner. "
        "Focus only on information directly relevant to the query, ignoring irrelevant details. "
        "Use numbered citations (e.g., [1]) after each key point or line, referencing the page number from the context. "
        "Do not include the raw context string or full source citations in your response. "
        "At the end, include a citation key listing all references (e.g., [1] = Page X of PDF Y)."
    )
    data = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": "You are an AI assistant for deep learning textbooks..."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "stream": True
    }
    try:
        response = requests.post(API_URL, json=data, headers=HEADERS, stream=True)
        if response.status_code != 200:
            return "Error contacting LLM."
        tokens = []
        for chunk in response.iter_lines():
            if chunk and chunk.startswith(b"data: "):
                chunk_data = chunk[6:].decode("utf-8")
                if chunk_data == "[DONE]":
                    break
                chunk_json = json.loads(chunk_data)
                token = chunk_json["choices"][0]["delta"].get("content", "")
                if token:
                    tokens.append(token)
        return "".join(tokens)
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return "Error generating response"
    





























@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[0], password):
            session["username"] = username
            logger.info(f"User {username} logged in")
            return redirect(url_for("dashboard"))
        logger.warning(f"Login failed for {username}")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                      (username, generate_password_hash(password)))
            conn.commit()
            os.makedirs(os.path.join("users", username), exist_ok=True)
            session["username"] = username
            logger.info(f"User {username} signed up")
            return redirect(url_for("dashboard"))
        except sqlite3.IntegrityError:
            conn.close()
            logger.warning(f"Signup failed: Username {username} already exists")
            return render_template("signup.html", error="Username already exists")
        finally:
            conn.close()
    return render_template("signup.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    
    username = session["username"]
    user_folder = os.path.join("users", username)
    pdfs = [f for f in os.listdir(user_folder) if f.endswith(".pdf")]
    
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(user_folder, filename)
                file.save(file_path)
                conn = sqlite3.connect("database.db")
                c = conn.cursor()
                c.execute("INSERT INTO tasks (username, step, progress, total, done) VALUES (?, ?, ?, ?, ?)",
                          (username, "Initializing", 0, 100, 0))
                task_id = c.lastrowid
                conn.commit()
                conn.close()
                logger.info(f"Starting PDF processing for {filename}, task_id={task_id}")
                asyncio.run_coroutine_threadsafe(process_pdf_folder(user_folder, username, task_id), async_loop)
                return jsonify({"message": "Processing started", "task_id": task_id})
        elif "delete" in request.form:
            pdf_to_delete = request.form["delete"]
            pdf_path = os.path.join(user_folder, pdf_to_delete)
            os.remove(pdf_path)
            try:
                collection = chroma_client.get_collection(name=f"{username}_embeddings")
                results = collection.get(where={"source": pdf_to_delete}, include=["metadatas"])
                for metadata in results["metadatas"]:
                    if metadata["type"] == "image" and os.path.exists(metadata["path"]):
                        os.remove(metadata["path"])
                collection.delete(where={"source": pdf_to_delete})
                logger.info(f"Deleted PDF {pdf_to_delete} and its embeddings")
            except Exception as e:
                logger.error(f"Error deleting embeddings/images for {pdf_to_delete}: {e}")
            return redirect(url_for("dashboard"))
    
    return render_template("dashboard.html", pdfs=pdfs, username=username)

@app.route("/progress")
def progress():
    if "username" not in session:
        return jsonify({"step": "Not started", "progress": 0, "done": False, "error": None, "task_id": None})
    username = session["username"]
    task_id = request.args.get("task_id", type=int)
    if not task_id:
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT task_id, step, progress, total, done, error FROM tasks WHERE username = ? AND done = 0 LIMIT 1", (username,))
        task = c.fetchone()
        conn.close()
        if task:
            task_id, step, progress, total, done, error = task
            return jsonify({"step": step, "progress": progress, "total": total, "done": bool(done), "error": error, "task_id": task_id})
        return jsonify({"step": "Not started", "progress": 0, "done": False, "error": "No task ID provided", "task_id": None})
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT step, progress, total, done, error FROM tasks WHERE username = ? AND task_id = ?",
              (username, task_id))
    task = c.fetchone()
    conn.close()
    
    if task:
        step, progress, total, done, error = task
        return jsonify({"step": step, "progress": progress, "total": total, "done": bool(done), "error": error, "task_id": task_id})
    return jsonify({"step": "Not started", "progress": 0, "done": False, "error": "Task not found", "task_id": task_id})

@app.route("/chat_progress")
def chat_progress():
    if "username" not in session:
        return jsonify({"step": "Not started", "progress": 0, "done": False, "response": None, "images": [], "error": None, "task_id": None})
    username = session["username"]
    task_id = request.args.get("task_id", type=int)
    if not task_id:
        return jsonify({"step": "Not started", "progress": 0, "done": False, "response": None, "images": [], "error": "No task ID provided", "task_id": None})
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT step, progress, total, done, response, images, error FROM chat_tasks WHERE username = ? AND task_id = ?",
              (username, task_id))
    task = c.fetchone()
    conn.close()
    
    if task:
        step, progress, total, done, response, images_json, error = task
        images = json.loads(images_json) if images_json else []
        logger.debug(f"Chat progress for task_id={task_id}: step={step}, progress={progress}, done={done}, response={response[:50] if response else None}, error={error}")
        return jsonify({"step": step, "progress": progress, "total": total, "done": bool(done), "response": response, "images": images, "error": error, "task_id": task_id})
    return jsonify({"step": "Not started", "progress": 0, "done": False, "response": None, "images": [], "error": "Task not found", "task_id": task_id})

@app.route("/chat", methods=["POST"])
def chat():
    if "username" not in session:
        return jsonify({"response": "Please log in.", "images": []})
    
    username = session["username"]
    data = request.json
    query = data["query"]
    mode = data.get("mode", "image+text")
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_tasks (username, query, mode, step, progress, total, done) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (username, query, mode, "Initializing", 0, 100, 0))
    task_id = c.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Chat request initiated: {username}, task_id={task_id}, query={query}, mode={mode}")
    asyncio.run_coroutine_threadsafe(process_chat_query(username, task_id, query, mode), async_loop)
    return jsonify({"message": "Chat processing started", "task_id": task_id})

@app.route("/logout")
def logout():
    username = session.pop("username", None)
    logger.info(f"User {username} logged out")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)