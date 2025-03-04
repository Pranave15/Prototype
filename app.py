import os
import PyPDF2
from flask import Flask, request, jsonify
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # UPDATED
from langchain_community.embeddings import SentenceTransformerEmbeddings  # UPDATED

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = None  # Store document embeddings

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def translate_to_english(text):
    detected_lang = detect(text[:500])
    if detected_lang in ["ta", "te", "kn", "hi"]:
        translator = GoogleTranslator(source=detected_lang, target="en")
        return translator.translate(text)
    return text

@app.route("/")  # FIX: Added a route for "/"
def home():
    return jsonify({"message": "Chatbot API is running!"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vector_db  # Store the processed document

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract and translate text
    extracted_text = extract_text_from_pdf(file_path)
    translated_text = translate_to_english(extracted_text)

    # Split and store embeddings
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(translated_text)
    vector_db = FAISS.from_texts(docs, embeddings)

    return jsonify({"message": "PDF uploaded and processed successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
