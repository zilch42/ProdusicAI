from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document  # Add this import
from logger import logger, log_function, log_rag_query
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

load_dotenv()

# TODO is it using rag effectively?

@log_function
def initialize_rag():
    """
    Initialize a RAG (Retrieval-Augmented Generation) system by creating a vector store from a CSV file.
    
    The function:
    1. Loads data from 'ideas.csv'
    2. Updates YouTube links for entries with songs
    3. Creates document embeddings using HuggingFace BGE embeddings
    4. Stores the embeddings in a Chroma vector store
    
    Returns:
        Chroma: A vector store containing the embedded documents
    """
    logger.info("Loading documents from ideas.csv")
    df = pd.read_csv("ideas.csv").replace({np.nan:None})
    df = update_youtube_links(df)
    df = df.drop(columns=['Timestamp'])

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
    vectorstore = Chroma.from_documents(documents, embedding_model)
    logger.info("Vector store creation complete")
    return vectorstore


def search_youtube_song(query: str, timestamp: str) -> str:
    """
    Search for a song on YouTube and return its URL with optional timestamp.
    
    Args:
        query (str): The search query for the song
        timestamp (str): Optional timestamp in 'MM:SS' format to append to the URL
        
    Returns:
        str: YouTube video URL with timestamp if provided, None if no video found
        
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

            if timestamp:
                try:
                    minutes, seconds = map(int, str(timestamp).split(':'))
                    t_seconds = minutes * 60 + seconds
                    video_url += f"&t={t_seconds}"
                except:
                    logger.warning(f"Invalid timestamp: {timestamp}")
                
            return video_url
        else:
            logger.warning(f"No YouTube videos found for query: {query}")
            return None
            
    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        raise
    finally:
        youtube.close()

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
    results = vectorstore.similarity_search(query, k=k)
    return results

# setup vectorstore
vectorstore = initialize_rag()