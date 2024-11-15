import os
from googleapiclient.discovery import build
from src.logger import logger
from src.config import get_config


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
        config = get_config()
        youtube = build('youtube', 'v3', developerKey=config.youtube_api_key, cache_discovery=False)
        
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
    

def create_youtube_embed(url: str, timestamp: str = "") -> str:
    """Create an HTML iframe for a YouTube URL."""
    video_id = url.split('=')[-1]
    html = f"""
<iframe 
    width="560" height="315" 
    src="https://www.youtube.com/embed/{video_id}?si=dTVYtHXBIJeY_EH-{timestamp}" 
    title="YouTube video player" frameborder="0" 
    allow="clipboard-write; encrypted-media; picture-in-picture" 
    referrerpolicy="no-referrer-when-downgrade" 
    allowfullscreen>
</iframe>
    """
    print(html)
    return html
