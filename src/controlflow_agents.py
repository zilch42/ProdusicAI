"""NOT IN USE
I got most of the way working through this branch. The AI agents were working, but I couldn't get the UI to work with the response without a time out from the javascript scroll lines. 
ControlFlow takes a lot longer than LangChain. Maybe 2-3x longer. It's much better from a coding experience perspective, but Cursor doesn't really know it at all so isn't much help. 
"""

import os
from typing import Any, Dict, List, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document  
from numpy import result_type
from pydantic import BaseModel
import controlflow as cf
from src.rag import get_random_by_category, get_ideas_db, search_youtube_song, DocMetadata
from src.logger import logger

# cf.settings.log_level = 'DEBUG'
# cf.settings.tools_verbose = True
cf.settings.enable_default_print_handler = False


class ConversationState(BaseModel):
    """Pydantic model for conversation state"""
    messages: List[BaseMessage]
    previous_messages: List[BaseMessage]
    current_agent: str = ""
    rag_results: List[Any] = []
    needs_song: bool = False
    suggested_song: Optional[str] = None
    youtube_url: Optional[str] = None
    is_random: bool = False
    is_followup: bool = False


def AzureLLM(temperature: float = 0.7) -> AzureChatOpenAI:
    """Initialize an Azure Chat OpenAI model"""
    try:
        llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
        temperature=temperature
    )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        logger.error("Please check your environment variables in `.env`")
        raise e
    return llm


# Create specialized agents
composer_agent = cf.Agent(
    name="Composer",
    description="Expert music composer and modern songwriter",
    instructions="You are an expert music composer and modern songwriter. Provide advice on composition, arrangement, and songwriting. Your specific influences include: Radiohead, Bloc Party, Sylvan Esso, Banks, As Tall as Lions, and Bon Iver.",
    model=AzureLLM(temperature=0.7)
)

mixing_engineer_agent = cf.Agent(
    name="Mixing Engineer",
    description="Expert mixing and mastering engineer",
    instructions="You are an expert mixing and mastering engineer. You studied under Nigel Godrich. Provide advice on mixing, EQ, compression, and other audio processing techniques.",
    model=AzureLLM(temperature=0.3)
)

sound_designer_agent = cf.Agent(
    name="Sound Designer",
    description="Expert sound designer and music producer",
    instructions="You are an expert sound designer and music producer. You studied under Nigel Godrich and Brian Eno. Provide advice on synthesizer programming, sample manipulation, and creating unique sounds.",
    model=AzureLLM(temperature=0.8)
)

lyricist_agent = cf.Agent(
    name="Lyricist",
    description="Expert lyricist",
    instructions="You are an expert lyricist. Provide advice on writing compelling lyrics, developing themes and narratives, crafting hooks, and ensuring lyrics flow well with the music. Your specific influences include: Radiohead, Bloc Party, Sylvan Esso, Banks, As Tall as Lions, and Bon Iver. You enjoy obscure and opaque lyrics, but not everything needs to be complicated.",
    model=AzureLLM(temperature=0.7)
)

song_suggestion_agent = cf.Agent(
    name="Song Suggestion",
    description="Expert song suggestion",
    instructions="You are a music clever with good technical knowledge and a wide range of musical tastes. Provide a good reference song that demonstrates the concept or technique discussed in the previous response.",
    model=AzureLLM(temperature=0.4), 
    tools=[search_youtube_song]
)

memory_agent = cf.Agent(
    name="Past Ideas",
    description="Musical idea database manager",
    instructions="You handle the user's self collected database of previous musical ideas. You can search this database for relevant information to help answer the user's current question. You will need to determine whether the ideas returned from the database are relevant to the user's current question.",
    model=AzureLLM(temperature=0.3)
)

coordinator = cf.Agent(
    name="Coordinator",
    instructions="""You are a project coordinator who routes queries and manages the conversation flow.
    Analyze the query and determine the next best action.""", 
    model=AzureLLM(temperature=0.4)
)


@cf.flow
async def music_production_workflow(message: str, previous_messages: Optional[List[BaseMessage]] = None) -> Dict:
    """Main workflow for handling music production queries"""
        
    # Task 1: Check for relevant RAG results
    rag_results = cf.run(
        "Get any relevant musical ideas stored in the database that are relevant to the user's current question",
        instructions="""You will need to determine whether the ideas returned from the database are relevant to the user's current question and filter the list appropriately.""",
        context=dict(query=message),
        result_type=list[DocMetadata],
        agents=[memory_agent],
        tools=[get_ideas_db]
    )

    # Task 2: Generate response
    specialist_responder = cf.Task(
        "Answer the user's query with the most appropriate specialist.",
        context=dict(query=message),
        result_type=str, 
        agents=[coordinator, composer_agent, mixing_engineer_agent, sound_designer_agent, lyricist_agent],
        completion_agents=[coordinator],   
    )
    specialist_response = specialist_responder.run(turn_strategy=cf.orchestration.turn_strategies.Moderated(moderator = coordinator))

    # Task 3: Generate song suggestion
    song_suggestion = cf.run(
        "Suggest a reference song that demonstrates the concept or technique discussed by the specialist agent",
        result_type=TypedDict('SongSuggestion', {'youtube_link': str, 'artist_song': str}),
        agents=[song_suggestion_agent],
        depends_on=[specialist_responder]
    )

    # Update state with results
    final_state = {
        "messages": specialist_response,
        "previous_messages": specialist_response, 
        "current_agent": "Unsure",
        "rag_results": rag_results,
        "suggested_song": song_suggestion.get("artist_song"),
        "youtube_url": song_suggestion.get("youtube_link"),
        "is_random": False,
        "is_followup": False
    }

    return final_state

# Keep the same interface for compatibility
async def invoke_agent(message: str, previous_messages: Optional[Sequence[BaseMessage]] = None) -> Dict:
    """Main entry point for the agent framework"""
    return await music_production_workflow(message, previous_messages)