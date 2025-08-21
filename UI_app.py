# <<<------------------------------------------------------------------>>>
#              || :::::  Import and Custom Class Setup   ::::: ||
# <<<------------------------------------------------------------------>>>
import os
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pymongo import MongoClient
import warnings
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import time
from pathlib import Path
import pyaudio
import wave
import numpy as np
import threading
from faster_whisper import WhisperModel

# Page config
st.set_page_config(page_title="Application System", page_icon="🛒", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .ai-response {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 20px 0;
        animation: fadeIn 0.5s;
    }
    
    /* Updated image styling for better fit */
    .stImage > img {
        max-width: 100%;
        max-height: 450px; /* Limit height */
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        padding: 10px;
        border-radius: 8px;
        margin: 0 auto;
        display: block;
    }
    
    /* Apply styling to the image container */
    .stImage {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        overflow: hidden; /* Prevent image overflow */
        display: flex;
        justify-content: center;
        align-items: center;
        height: auto; /* Auto height */
    }
    
    /* Caption styling */
    .image-caption {
        text-align: center;
        padding: 8px 5px;
        font-size: 0.9rem;
        color: #666;
        background-color: #f9f9f9;
        border-radius: 0 0 8px 8px;
        margin-top: 5px;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    
    /* Audio recording button styling */
    .recording-button {
        background-color: #DC2626 !important;
    }
    
    /* Adjust the caption container */
    .css-1kyxreq {
        margin-top: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")

# App title
st.markdown("<h1 class='main-header'>Application System</h1>", unsafe_allow_html=True)


# Audio Transcription Class
class LiveAudioTranscriber:
    def __init__(self, model_size="medium", device="auto"):
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.recording = False
        self.frames = []

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Initialize Whisper model
        self.model = WhisperModel(model_size, device=device)

        # Temporary file path
        self.temp_file = "temp_recording.wav"

    def start_recording(self):
        self.recording = True
        self.frames = []

        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()

    def _record(self):
        while self.recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def stop_recording(self):
        if not self.recording:
            return "Not currently recording."

        self.recording = False
        self.record_thread.join()

        # Close and terminate audio stream
        self.stream.stop_stream()
        self.stream.close()

        # Save recording to temporary file
        self._save_audio()

        # Transcribe the audio
        result = self._transcribe()

        # Clean up temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

        return result

    def _save_audio(self):
        wf = wave.open(self.temp_file, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(self.frames))
        wf.close()

    def _transcribe(self):
        segments, info = self.model.transcribe(self.temp_file, beam_size=5)

        # Collect transcription text
        text = " ".join([segment.text for segment in segments])
        return text

    def close(self):
        """Clean up resources"""
        self.audio.terminate()


# <<<------------------------------------------------------------------>>>
#              || :::::  Step 0 :Database Initialization  ::::: ||
# <<<------------------------------------------------------------------>>>
@st.cache_resource
def initialize_databases():
    # MongoDB connection
    client = MongoClient("mongodb://localhost:27017/")
    db = client["E_Commerce"]
    mongo_collection = db["samples"]

    # Create directory to save images
    dataset_folder = "./dataset/mongodb_images"
    os.makedirs(dataset_folder, exist_ok=True)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./data/e_commerce.db")
    image_loader = ImageLoader()
    embedding_function = OpenCLIPEmbeddingFunction()

    # Create separate collections for text and image embeddings
    image_collection = chroma_client.get_or_create_collection(
        "e_commerce_images",
        embedding_function=embedding_function,
        data_loader=image_loader,
    )

    text_collection = chroma_client.get_or_create_collection(
        "e_commerce_text",
        embedding_function=embedding_function,
    )

    return mongo_collection, image_collection, text_collection, dataset_folder


# Initialize connections
mongo_collection, image_collection, text_collection, dataset_folder = initialize_databases()


# Initialize audio transcriber in session state if not already there
if "transcriber" not in st.session_state:
    st.session_state.transcriber = LiveAudioTranscriber(model_size="medium")
    st.session_state.recording = False


# Functions for data processing
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Failed to download image, status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return False


def save_images_from_mongodb():
    # Path to track which images we've already downloaded
    download_tracker = os.path.join(dataset_folder, "downloaded_ids.json")

    # Load previously downloaded image IDs
    downloaded_ids = set()
    if os.path.exists(download_tracker):
        try:
            with open(download_tracker, "r") as f:
                downloaded_ids = set(json.load(f))
        except Exception as e:
            st.error(f"Error loading download tracker: {e}")

    # Query documents from MongoDB
    documents = mongo_collection.find().limit(200)

    count = 0
    new_count = 0
    skipped_count = 0

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Keep track of IDs for this run
    current_ids = set()
    total_docs = mongo_collection.count_documents({})

    for i, doc in enumerate(documents):
        progress_bar.progress((i + 1) / min(200, total_docs))
        status_text.text(f"Processing document {i + 1}/{min(200, total_docs)}")

        doc_id = str(doc.get("_id", ""))
        current_ids.add(doc_id)

        # Skip if already downloaded
        if doc_id in downloaded_ids:
            skipped_count += 1
            continue

        if "image" in doc and doc["image"]:
            image_url = doc["image"]
            # Extract filename from ObjectId or create a sequential one
            filename = f"{doc_id}_{doc.get('title', '')}.jpg"
            # Clean filename of any invalid characters
            filename = "".join(c for c in filename if c.isalnum() or c in ["_", "-", "."]).strip()

            save_path = os.path.join(dataset_folder, filename)

            status_text.text(f"Downloading image {i + 1}: {image_url}")
            success = download_image(image_url, save_path)

            if success:
                count += 1
                new_count += 1
                downloaded_ids.add(doc_id)

    # Update our tracker file with all downloaded IDs
    with open(download_tracker, "w") as f:
        json.dump(list(downloaded_ids), f)

    progress_bar.empty()
    status_text.empty()

    return new_count, skipped_count, len(downloaded_ids)


def update_chromadb():
    # Track which files are already in ChromaDB
    chromadb_tracker = os.path.join(dataset_folder, "chromadb_ids.json")
    chromadb_ids = set()

    # Load previously added ChromaDB IDs
    if os.path.exists(chromadb_tracker):
        try:
            with open(chromadb_tracker, "r") as f:
                chromadb_ids = set(json.load(f))
        except Exception as e:
            st.error(f"Error loading ChromaDB tracker: {e}")

    # Get current counts
    existing_img_count = image_collection.count()
    existing_txt_count = text_collection.count()

    # Prepare new data to add
    new_img_ids, new_img_uris = [], []
    new_txt_ids, new_txt_data, new_txt_metadata = [], [], []
    added_ids = []

    # Get MongoDB documents for text data
    mongo_docs = {str(doc['_id']): doc for doc in mongo_collection.find()}

    for filename in sorted(os.listdir(dataset_folder)):
        if filename.endswith((".jpg", ".png")) and not filename.startswith("."):
            # Extract MongoDB ID from filename
            mongo_id = filename.split("_")[0] if "_" in filename else filename.split(".")[0]
            
            # Skip if already processed
            if mongo_id in chromadb_ids:
                continue

            file_path = os.path.join(dataset_folder, filename)
            
            # Add image embedding
            new_img_ids.append(f"{mongo_id}_img")
            new_img_uris.append(file_path)
            
            # Add text embedding if MongoDB document exists
            if mongo_id in mongo_docs:
                doc = mongo_docs[mongo_id]
                text_content = f"{doc.get('title', '')} {doc.get('description', '')} {doc.get('category', '')}".strip()
                
                if text_content:
                    new_txt_ids.append(f"{mongo_id}_txt")
                    new_txt_data.append(text_content)
                    new_txt_metadata.append({
                        'mongo_id': mongo_id,
                        'title': doc.get('title', ''),
                        'description': doc.get('description', ''),
                        'category': doc.get('category', ''),
                        'type': 'text'
                    })
            
            added_ids.append(mongo_id)

    # Add to ChromaDB collections
    total_added = 0
    if new_img_ids:
        with st.spinner(f"Adding {len(new_img_ids)} images to ChromaDB"):
            image_collection.add(
                ids=new_img_ids, 
                uris=new_img_uris,
                metadatas=[{'mongo_id': id.split('_')[0], 'type': 'image'} for id in new_img_ids]
            )
        total_added += len(new_img_ids)

    if new_txt_ids:
        with st.spinner(f"Adding {len(new_txt_ids)} text embeddings to ChromaDB"):
            text_collection.add(
                ids=new_txt_ids,
                documents=new_txt_data,
                metadatas=new_txt_metadata
            )
        total_added += len(new_txt_ids)

    # Update tracker
    if added_ids:
        chromadb_ids.update(added_ids)
        with open(chromadb_tracker, "w") as f:
            json.dump(list(chromadb_ids), f)

    return total_added, image_collection.count(), text_collection.count(), len(chromadb_ids)


# <<<------------------------------------------------------------------>>>
#              || :::::  Step 2 Semantic Search  ::::: ||
# <<<------------------------------------------------------------------>>>


def query_db(query, results=4):
    with st.spinner("Searching for relevant images and text..."):
        # Search in both collections
        img_results = image_collection.query(
            query_texts=[query], 
            n_results=results, 
            include=["uris", "distances", "metadatas"]
        )
        
        txt_results = text_collection.query(
            query_texts=[query], 
            n_results=results, 
            include=["documents", "distances", "metadatas"]
        )
        
        # Combine and deduplicate results by mongo_id
        combined_results = {}
        
        # Process image results
        for i, (uri, distance, metadata) in enumerate(zip(
            img_results["uris"][0], 
            img_results["distances"][0], 
            img_results["metadatas"][0]
        )):
            mongo_id = metadata['mongo_id']
            if mongo_id not in combined_results or distance < combined_results[mongo_id]['distance']:
                combined_results[mongo_id] = {
                    'uri': uri,
                    'distance': distance,
                    'type': 'image',
                    'metadata': metadata
                }
        
        # Process text results
        for i, (doc, distance, metadata) in enumerate(zip(
            txt_results["documents"][0] if txt_results["documents"] else [],
            txt_results["distances"][0], 
            txt_results["metadatas"][0]
        )):
            mongo_id = metadata['mongo_id']
            # Weight text results slightly lower (multiply by 1.1)
            adjusted_distance = distance * 1.1
            
            if mongo_id not in combined_results or adjusted_distance < combined_results[mongo_id]['distance']:
                # Need to find corresponding image URI
                img_path = None
                for filename in os.listdir(dataset_folder):
                    if filename.startswith(mongo_id) and filename.endswith(('.jpg', '.png')):
                        img_path = os.path.join(dataset_folder, filename)
                        break
                
                combined_results[mongo_id] = {
                    'uri': img_path,
                    'distance': adjusted_distance,
                    'type': 'text',
                    'metadata': metadata,
                    'document': doc
                }
        
        # Sort by distance and convert back to expected format
        sorted_results = sorted(combined_results.values(), key=lambda x: x['distance'])[:results]
        
        return {
            "uris": [[r['uri'] for r in sorted_results if r['uri']]],
            "distances": [[r['distance'] for r in sorted_results if r['uri']]],
            "metadatas": [[r['metadata'] for r in sorted_results if r['uri']]]
        }


# <<<------------------------------------------------------------------>>>
#              || :::::  Step 3 Relevant Images  ::::: ||
# <<<------------------------------------------------------------------>>>


def format_prompt_inputs(data, user_query):
    with st.spinner("Processing images for AI analysis..."):
        inputs = {}
        inputs["user_query"] = user_query

        # Collect relevant paths
        relevant = []
        for path, score in zip(data["uris"][0], data["distances"][0]):
            if score >= 0.90:  # adjust threshold
                relevant.append(path)

        # Convert strings to Path objects
        relevant_paths = [Path(p) for p in relevant]

        # Just use them directly (no nested loop)
        image_paths = list(relevant_paths)

        # Take up to 4 images safely
        encoded_images = []
        for i, image_path in enumerate(image_paths[:4]):
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            encoded_images.append(base64.b64encode(image_data).decode("utf-8"))
            inputs[f"image_data_{i + 1}"] = encoded_images[-1]

        return inputs, image_paths


# def format_prompt_inputs(data, user_query):
#     with st.spinner("Processing images for AI analysis..."):
#         inputs = {"user_query": user_query}


#         # Collect all image paths from ChromaDB
#         image_paths = []
#         for group in data["uris"]:  # Each group is a list of image paths
#             image_paths.extend(group)

#         # Encode all images once
#         encoded_images = []
#         for image_path in image_paths:
#             try:
#                 with open(image_path, "rb") as image_file:
#                     image_data = image_file.read()
#                 encoded_images.append(base64.b64encode(image_data).decode("utf-8"))
#             except Exception as e:
#                 print(f"Error reading {image_path}: {e}")
#                 encoded_images.append(None)

#         # 🔹 Batch filtering with encoded images + paths
#         relevant_paths = filter_relevant_images(user_query, encoded_images, image_paths)

#         # Take up to 4 relevant images safely
#         for i, image_path in enumerate(relevant_paths[:4]):
#             with open(image_path, "rb") as image_file:
#                 image_data = image_file.read()
#             inputs[f"image_data_{i + 1}"] = base64.b64encode(image_data).decode("utf-8")

#         return inputs, relevant_paths


# def filter_relevant_images(
#     user_query: str, encoded_images: list[str], image_paths: list[str]
# ) -> list[str]:
#     """
#     Given a list of base64-encoded images and their paths,
#     return only the paths of relevant images.
#     """
#     vision_model = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash-latest",
#         temperature=0.0,
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#     )
#     parser = StrOutputParser()

#     # Construct multi-image message
#     images_content = []
#     for i, img_b64 in enumerate(encoded_images):
#         images_content.append(
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
#         )

#     messages = [
#         SystemMessage(
#             content="You are a philatelist assistant. Return a comma-separated list of image indices (starting at 1) that match the query."
#         ),
#         HumanMessage(
#             content=[
#                 {
#                     "type": "text",
#                     "text": f"User query: '{user_query}'. Which of these images are relevant?",
#                 },
#                 *images_content,
#             ]
#         ),
#     ]

#     try:
#         response = vision_model.invoke(messages)
#         parsed_response = parser.invoke(response).strip()

#         # Example response: "1,3" → means images 1 and 3 are relevant
#         relevant_indices = [
#             int(x.strip()) - 1 for x in parsed_response.split(",") if x.strip().isdigit()
#         ]

#         return [image_paths[i] for i in relevant_indices if i < len(image_paths)]
#     except Exception as e:
#         print(f"Error filtering relevance: {e}")
#         return []


# Function to resize and normalize images for better display
def preprocess_image_for_display(image_path):
    try:
        # Open the image
        img = Image.open(image_path)

        # Calculate aspect ratio
        aspect_ratio = img.width / img.height

        # Determine new dimensions while preserving aspect ratio
        max_width = 600  # Maximum width for display
        max_height = 400  # Maximum height for display

        if aspect_ratio > 1:  # Wider than tall
            new_width = min(img.width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:  # Taller than wide
            new_height = min(img.height, max_height)
            new_width = int(new_height * aspect_ratio)

        # Resize the image if needed
        if img.width > max_width or img.height > max_height:
            img = img.resize((new_width, new_height), Image.LANCZOS)

        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return Image.open(image_path)  # Return original if processing fails


# Function to start recording
def start_recording():
    st.session_state.recording = True
    st.session_state.transcriber.start_recording()
    st.rerun()


# Function to stop recording
def stop_recording():
    st.session_state.recording = False
    transcribed_text = st.session_state.transcriber.stop_recording()
    # Set the transcribed text as the query
    st.session_state.query_input = transcribed_text
    st.rerun()


# App workflow
with st.sidebar:
    st.markdown("<h2 class='sub-header'>System Status</h2>", unsafe_allow_html=True)

    # Data refresh button
    if st.button("Refresh Data"):
        with st.spinner("Processing images from MongoDB..."):
            new_count, skipped_count, total_count = save_images_from_mongodb()
            st.success(f"Downloaded {new_count} new images")
            st.info(f"Skipped {skipped_count} existing images")
            st.info(f"Total images in dataset: {total_count}")

        with st.spinner("Updating ChromaDB..."):
            total_added, img_count, txt_count, total_tracked = update_chromadb()
            if total_added > 0:
                st.success(f"Added {total_added} new embeddings to ChromaDB")
            else:
                st.info("No new data to add to ChromaDB")
            st.info(f"Images in ChromaDB: {img_count}")
            st.info(f"Text entries in ChromaDB: {txt_count}")

    # Show system information
    st.markdown("### System Information")
    st.markdown("- Application: Streamlit + Python")
    st.markdown("- Database: MongoDB + ChromaDB")
    st.markdown("- GEN AI Model: Gemini")
    st.markdown("- Image Embedding: OpenCLIP")

# Add text input with previous transcription if available
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input(
    "Advance Search",
    value=st.session_state.query_input,
    key="query_input",
)

# Create a layout for the input options
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

with col2:
    # Create a button to start/stop recording
    if not st.session_state.recording:
        st.button("🎤 Voice Search", on_click=start_recording, use_container_width=True)
    else:
        st.button(
            "⏹️ Stop Recording",
            on_click=stop_recording,
            use_container_width=True,
            help="Stop recording and transcribe",
        )

with col3:
    search_button = st.button("🔍 Search", use_container_width=True)

# Display recording status
if st.session_state.recording:
    st.info("🎙️ Recording... Speak clearly and then click 'Stop Recording' when finished.")

# <<<------------------------------------------------------------------>>>
#              || :::::  Step 1 Advance Search  ::::: ||
# <<<------------------------------------------------------------------>>>

if search_button and query:
    # Query processing
    results = query_db(query)

    if len(results["uris"][0]) >= 1:
        # Get prompt inputs and all image paths
        prompt_inputs, image_paths = format_prompt_inputs(results, query)

        response_container = st.empty()

        # Display related images
        st.markdown("<h2 class='sub-header'>Relevant Stamps</h2>", unsafe_allow_html=True)

        # Create a grid layout for displaying multiple images
        cols = st.columns(min(4, max(1, len(image_paths))))  # up to 4 images per row, minimum 1

        for i, path in enumerate(image_paths):
            with cols[i % 4]:  # distribute images across columns
                with st.container():
                    img = preprocess_image_for_display(path)
                    st.image(img, caption=f"Stamp Sample {i + 1}", use_container_width=True)

    else:
        st.error("Not enough relevant images found. Please try a different query.")
elif search_button:
    st.warning("Please enter a query or use voice search to find stamps.")


# Handle app close
def on_close():
    if "transcriber" in st.session_state:
        st.session_state.transcriber.close()


# Register the on_close function to be called when the app is closed
try:
    # This is a workaround as Streamlit doesn't have a built-in on_close event
    st.cache_resource.clear()
except:
    pass

# Footer
st.markdown("---")
st.markdown("© 2025  Application System - Powered by AI")
