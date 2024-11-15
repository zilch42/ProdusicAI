import random
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document  
from googleapiclient.discovery import build
import os
import pandas as pd
import numpy as np
import hashlib
import asyncio
import shutil
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from src.logger import logger, log_function, log_rag_query

@log_function
def get_csv_hash():
    """
    Calculate SHA-256 hash of ideas.csv file.
    """
    with open("ideas.csv", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@log_function
async def initialize_rag():
    """
    Initialize a RAG (Retrieval-Augmented Generation) system.

    This function either loads a cached vector store if available and the source CSV hasn't changed,
    or creates a new vector store from the ideas.csv file.

    Returns:
        Chroma: Initialized vector store instance

    Side Effects:
        - Creates/updates chroma_db directory
        - Creates/updates ideas_csv.hash file
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
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # If we reach here, we need to create a new database
    logger.info("Creating new vector store from ideas.csv")
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    df = pd.read_csv("ideas.csv").replace({np.nan:None})
    df = await update_youtube_links(df)

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
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Creating vector store from documents")
    vectorstore = Chroma.from_documents(
        documents, 
        embedding_model, 
        persist_directory=persist_directory
    )
    
    # Save the hash
    with open(hash_file, "w") as f:
        f.write(csv_hash)
    
    logger.info("Vector store creation complete")
    return vectorstore


async def search_youtube_song(query: str) -> str:
    """
    Search for a song on YouTube and return its URL.
    
    Args:
        query (str): The search query for the song (e.g. Artist - Title)
        
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
async def update_youtube_links(df):
    """
    Update YouTube links in the DataFrame for entries that have songs but no links.
    
    The function handles both single songs and lists of songs, matching songs with 
    their corresponding timestamps if available.
    
    Args:
        df (pandas.DataFrame): DataFrame containing song information with columns:
            - Song: String or list of songs
            - Link: Existing YouTube links (if any)
            - Timestamp: Optional timestamps for the songs
            
    Returns:
        pandas.DataFrame: Updated DataFrame with new YouTube links
        
    Side Effects:
        - Updates ideas.csv with new YouTube links
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
                    link = await search_youtube_song(song)
                    logger.info(f"Found YouTube link for {song}")
                    links.append(link)
                record.Link = str(links)
                continue
            else:
                record.Link = await search_youtube_song(record.Song)
            logger.info(f"Found YouTube link for {record.Song}")

    # Save the updated DataFrame back to CSV
    df.to_csv("ideas.csv", index=False)
    logger.info("Saved updated ideas.csv")

    return df 


class DocMetadata(BaseModel):
    Category: str
    Technique: str
    Description: str
    Song: Optional[str] = None
    Link: Optional[str] = None
    Timestamp: Optional[str] = None


@log_rag_query
@log_function
async def get_ideas_db(query, k=3) -> list[DocMetadata]:
    """
    Query the RAG db of musical ideas to find similar documents based on semantic search.

    Args:
        query (str): The search query 
        k (int, optional): Number of results to return. Defaults to 3.

    Returns:
        list[DocMetadata]: List of matching documents
    """
    global _vectorstore
    results = await _vectorstore.asimilarity_search(query, k=k)
    return [DocMetadata(**result.metadata) for result in results]

@log_function
async def get_random_by_category(category: str = None) -> None | DocMetadata:
    """
    Get a random document from the RAG system matching a specific category.
    
    Args:
        category (str | list[str] | None): Category or list of categories to filter by.
            If None, selects from all categories.
        
    Returns:
        DocMetadata | None: A random document from the specified category/categories,
            or None if no matches found
    """
    # For filtering by single category or list of categories
    where = {"Category": {"$in": [category]} if isinstance(category, str) else {"$in": category}} if category else None

    global _vectorstore
    ids = _vectorstore.get(where=where, include=['documents']).get('ids', [])
    if not ids:
        logger.warning(f"No documents found for category: {category}")
        return None
    
    random_id = random.choice(ids)
    record = _vectorstore.get(ids=random_id)
    return DocMetadata(**record['metadatas'][0])

async def get_category_list():
    """
    Get a unique list of all categories from the vector store metadata.

    Returns:
        list[str]: Sorted list of unique categories
    """
    global _vectorstore
    results = _vectorstore.get(include=['metadatas'])
    categories = {doc.get('Category') for doc in results['metadatas'] if doc.get('Category')}
    return sorted(list(categories))


# Initialize the vector store when the module loads
logger.info("Initializing vector store on startup")
_vectorstore = asyncio.run(initialize_rag())
_rag_categories = asyncio.run(get_category_list())