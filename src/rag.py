from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document  
from logger import logger, log_function, log_rag_query
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import hashlib

load_dotenv()

@log_function
def get_csv_hash():
    """Calculate SHA-256 hash of ideas.csv"""
    with open("ideas.csv", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

@log_function
def initialize_rag():
    """
    Initialize a RAG system, using cached vector store if available and CSV hasn't changed.
    """
    csv_hash = get_csv_hash()
    persist_directory = "chroma_db"
    hash_file = "ideas_csv.hash"

    # Check if we can use cached database
    if os.path.exists(persist_directory) and os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == csv_hash:
            logger.info("Loading vector store from disk cache")
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name="BAAI/llm-embedder",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
            return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # If we reach here, we need to create a new database
    logger.info("Creating new vector store from ideas.csv")
    df = pd.read_csv("ideas.csv").replace({np.nan:None})
    df = update_youtube_links(df)

    # Text to be embedded 
    # Combine relevant columns into text for embedding
    documents = []
    for idx, row in df.iterrows():
        text = f"Category: {row['Category']}\nTechnique: {row['Technique']}\nDescription: {row['Description']}"
        metadata = row.to_dict()
        metadata = {k: v for k, v in metadata.items() if v is not None}
        documents.append(Document(page_content=text, metadata=metadata))

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/llm-embedder",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Creating vector store from documents")
    vectorstore = Chroma.from_documents(
        documents, 
        embedding_model, 
        persist_directory=persist_directory
    )
    vectorstore.persist()
    
    # Save the hash
    with open(hash_file, "w") as f:
        f.write(csv_hash)
    
    logger.info("Vector store creation complete")
    return vectorstore


def search_youtube_song(query: str) -> str:
    """
    Search for a song on YouTube and return its URL.
    
    Args:
        query (str): The search query for the song
        
    Returns:
        str: YouTube video URL, None if no video found
        
    Raises:
        Exception: If there's an error during the YouTube API request
    """
    try:
        youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"), cache_discovery=False)
        
        if "remix" not in query.lower():
            query = f"{query} official"

        # Perform the search
        logger.info(f"Searching YouTube for: {query}")
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=1,
            type='video'
        ).execute()
        
        # Get the video ID from the response
        if search_response['items']:
            video_id = search_response['items'][0]['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Found YouTube video: {video_url}")                
            return video_url
        else:
            logger.warning(f"No YouTube videos found for query: {query}")
            return None
            
    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        raise
    finally:
        youtube.close()

def convert_timestamp_to_yt(timestamp: str) -> str:
    """
    Convert a timestamp string to a URL parameter for YouTube.
    """
    try:
        minutes, seconds = map(int, str(timestamp).split(':'))
        t_seconds = minutes * 60 + seconds
        ts = f"&amp;start={t_seconds}"
        return ts
    except:
        logger.warning(f"Invalid timestamp: {timestamp}")
        return ""

@log_function
def update_youtube_links(df):
    """
    Update YouTube links in the DataFrame for entries that have songs but no links.
    
    The function handles both single songs and lists of songs. For lists, it matches
    songs with their corresponding timestamps if available.
    
    Args:
        df (pandas.DataFrame): DataFrame containing song information with columns:
            - Song: String or list of songs
            - Link: Existing YouTube links (if any)
            - Timestamp: Optional timestamps for the songs
            
    Returns:
        pandas.DataFrame: Updated DataFrame with new YouTube links
        
    Side effects:
        - Saves the updated DataFrame back to 'ideas.csv'
        - Logs the progress of finding YouTube links
    """

    for record in df.itertuples():
        if record.Song and not record.Link:
            # Handle case where Song is a list of songs
            if isinstance(record.Song, str) and record.Song.startswith('['):
                # Convert string representation of list to actual list
                songs = eval(record.Song)
                if isinstance(record.Timestamp, str) and record.Timestamp.startswith('['):
                    timestamps = eval(record.Timestamp)
                else:
                    timestamps = [None]*len(songs)    
                links = []
                for song, timestamp in zip(songs, timestamps):
                    link = search_youtube_song(song, timestamp)
                    logger.info(f"Found YouTube link for {song}")
                    links.append(link)
                record.Link = str(links)
                continue
            else:
                record.Link = search_youtube_song(record.Song, record.Timestamp)
            logger.info(f"Found YouTube link for {record.Song}")

    # Save the updated DataFrame back to CSV
    df.to_csv("ideas.csv", index=False)
    logger.info("Saved updated ideas.csv")

    return df 

@log_rag_query
@log_function
def query_rag(query, k=3):
    """
    Query the RAG system to find similar documents based on semantic search.
    """
    global _vectorstore
    results = _vectorstore.similarity_search(query, k=k)
    return results

# Initialize the vector store when the module loads
logger.info("Initializing vector store on startup")
_vectorstore = initialize_rag()
