from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)


# Combined specialist definitions with their temperatures and system messages
SPECIALISTS = {
    "composer": {
        "temperature": 0.7,  # More creative for composition suggestions
        "system_message": SystemMessage(content="You are an expert music composer. Provide advice on composition, arrangement, and songwriting.")
    },
    "mixing_engineer": {
        "temperature": 0.3,  # More precise for technical mixing advice
        "system_message": SystemMessage(content="You are an expert mixing engineer. Provide advice on mixing, EQ, compression, and other audio processing techniques.")
    },
    "sound_designer": {
        "temperature": 0.8,  # Very creative for sound design ideas
        "system_message": SystemMessage(content="You are an expert sound designer. Provide advice on synthesizer programming, sample manipulation, and creating unique sounds.")
    },
    "lyricist": {
        "temperature": 0.7,  # Creative but structured for lyric writing
        "system_message": SystemMessage(content="You are an expert lyricist and songwriter. Provide advice on writing compelling lyrics, developing themes and narratives, crafting hooks, and ensuring lyrics flow well with the music.")
    },
    "project_manager": {
        "temperature": 0.4,  # More focused for project coordination
        "system_message": SystemMessage(content="You are a music project manager. Coordinate the efforts and provide overall guidance on the music production process.")
    }
}

def get_specialist(query):
    """Determine which specialist should handle the query"""
    routing_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a project coordinator. Your job is to route queries to the appropriate specialist."),
        HumanMessage(content=f"""Based on this query: "{query}"
        
        Determine which specialist would be most appropriate:
        - composer: For composition, arrangement, and music theory
        - mixing_engineer: For audio processing, mixing, mastering and sound balancing
        - sound_designer: For audio synthesis, sound creation, and sonic textures
        - lyricist: For writing lyrics, themes, hooks and song narratives
        - project_manager: For project coordination or unclear queries
        
        Respond only with the role key, nothing else.""")
    ])
    
    response = llm.invoke(routing_prompt.messages)
    specialist = response.content.lower().strip()
    return specialist if specialist in SPECIALISTS else "project_manager"
