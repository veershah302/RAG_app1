import os
# pip install fitz
import fitz
from PIL import Image

import io
import torch
# pip install transformers
from transformers import CLIPProcessor, CLIPModel
# pip install PyPDF2
from PyPDF2 import PdfReader
# pip install chromadb
import chromadb
import numpy as np
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 200
BATCH_SIZE = 5000

def extract_text_and_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    text_by_page = {}
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text() or ""
        text_by_page[page_num + 1] = text
        
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_path = os.path.join(output_folder, f"{os.path.basename(pdf_path)}_p{page_num+1}_i{img_idx+1}.{image_ext}")
            image.save(image_path)
            image_paths.append((image_path, page_num + 1))
    
    return text_by_page, image_paths

def chunk_text(text, chunk_size=CHUNK_SIZE):
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
    return chunks

def get_clip_embedding(content, is_image=False):
    try:
        if is_image:
            image = Image.open(content).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
        else:
            inputs = clip_processor(text=[content], return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs) if is_image else clip_model.get_text_features(**inputs)
        return embedding.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def add_to_chroma_in_batches(collection, embeddings, metadatas, ids, status_dict):
    total = len(embeddings)
    status_dict["total"] = total
    for i in range(0, total, BATCH_SIZE):
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        collection.add(embeddings=batch_embeddings, metadatas=batch_metadatas, ids=batch_ids)
        status_dict["progress"] = min(i + BATCH_SIZE, total)
        print(f"Processed {status_dict['progress']}/{total} items")

def process_pdf_folder(pdf_folder, username, status_dict):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        status_dict["done"] = True
        return
    
    try:
        collection = chroma_client.get_collection(name=f"{username}_embeddings")
        chroma_client.delete_collection(name=f"{username}_embeddings")
    except:
        pass
    collection = chroma_client.create_collection(name=f"{username}_embeddings", metadata={"dimension": 512})
    
    embeddings = []
    metadatas = []
    next_id = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text_by_page, image_paths = extract_text_and_images_from_pdf(pdf_path, pdf_folder)
        
        for page_num, text in text_by_page.items():
            chunks = chunk_text(text)
            for chunk in chunks:
                embedding = get_clip_embedding(chunk)
                if embedding and len(embedding) > 0:
                    embeddings.append(embedding)
                    metadatas.append({"type": "text", "content": chunk, "source": pdf_file, "page": page_num})
        
        for image_path, page_num in image_paths:
            embedding = get_clip_embedding(image_path, is_image=True)
            if embedding and len(embedding) > 0:
                description = text_by_page.get(page_num, "Image from textbook")[-200:]
                embeddings.append(embedding)
                metadatas.append({"type": "image", "path": image_path, "description": description, "source": pdf_file, "page": page_num})
    
    if embeddings:
        ids = [str(i) for i in range(next_id, next_id + len(embeddings))]
        add_to_chroma_in_batches(collection, embeddings, metadatas, ids, status_dict)
    status_dict["done"] = True