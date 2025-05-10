Multimodal PDF Chat Application
A Flask-based web app that enables users to query academic PDFs using text, images, or both. Built using CLIP, BLIP, and ChromaDB, it supports LLM-based reasoning via llamafile for summarization or alternatively using OpenAI API.
################################################################################
Features

Upload PDFs and extract both text & images

Generate CLIP embeddings for semantic search

Caption diagrams using BLIP and store embeddings

Chat interface supporting:

Text queries (e.g. “Explain transformers”)

Image queries (e.g. “Show CNN diagrams”)

Hybrid queries (text + image)

generate LLM-based answers with citations

User-friendly web interface (Flask)
################################################################################
Tech Stack
PyMuPDF - PDF text/image extraction
CLIP - Text & image embeddings
BLIP - Image captioning
ChromaDB - Vector similarity search
Flask - Web frontend & backend
llamafile - Optional LLM summarization backend
SQLite - User & task storage
#################################################################################
Getting Started


1. Create a Virtual Environment
(terminal) 
Run:
python -m venv venv

Then activate:
On Linux/Mac: source venv/bin/activate
On Windows: venv\Scripts\activate

2. Install Python Dependencies

Run:
pip install -r requirements.txt

3. Download llamafile (Optional for LLM support)

Go to https://github.com/Mozilla-Ocho/llamafile

Download the LLaMA 3.2 3B Instruct version for your system (we tested this on Windows)

You can pick a version based on your OS (Linux/Mac/Windows) and hardware (CPU or GPU)

The method of running the file will depend on this:

Windows: use .exe

Linux/Mac: run as ./llamafile

Rename and Place llamafile

Rename the downloaded file:
From: Llama-3.2-3B-Instruct.Q6_K.llamafile
To: Llama-3.2-3B-Instruct.Q6_K.exe

Place the renamed file (.exe) in the root of the project folder

(Optional) Use OpenAI Instead of llamafile

You can skip llamafile completely and instead use an OpenAI API key

To do this, open app.py and update the LLM endpoint configuration accordingly

Run the LLM Backend (if using llamafile)

In a separate terminal, run:
./Llama-3.2-3B-Instruct.Q6_K.exe --model models/your_model.gguf

4. Run the Application

In your main terminal, run:
python app.py

Open the Web Interface

After the app starts, open your browser and go to the address shown in your terminal
(typically http://127.0.0.1:5000 or http://localhost:5000)

##############################################################################################################
Folder Structure
.
├── app.py # Main Flask app with routes
├── preprocessor.py # PDF preprocessing utilities
├── templates/ # HTML files
├── static/ # Frontend assets
├── chroma_db/ # Vector DB files
├── users/ # Uploaded files & images
├── Llama-3.2-3B-Instruct.Q6_K.exe # llamafile binary (optional)
└── _Technical Report.docx # Project summary

Highlighting the sample images, depicting functionality:
![signing up](<WhatsApp Image 2025-04-09 at 14.33.37_bafd46c9.jpg>)
![Login](<WhatsApp Image 2025-04-09 at 14.33.27_3aba67d0.jpg>)  
![Uploading PDF's of Backpropagation and Activation Functions](<WhatsApp Image 2025-04-09 at 14.34.15_24fe88ae.jpg>)
![Picture Identification- Sir](<WhatsApp Image 2025-04-09 at 14.33.04_23d21a83.jpg>)
![ONLY IMAGE: Asking about ReLU Activation Function](<WhatsApp Image 2025-04-09 at 14.30.44_fdd1513a.jpg>)
![IMAGE and TEXT: Asking about ReLU Activation Function](<WhatsApp Image 2025-04-09 at 14.31.34_9332ed91.jpg>)
![TEXT- Asking about Backpropogation](<WhatsApp Image 2025-04-09 at 14.32.32_b041c75a.jpg>)
