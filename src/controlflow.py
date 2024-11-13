import os
from typing import Any, Dict, List, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
import controlflow as cf
from src.rag import get_random_by_category, query_rag, search_youtube_song, _rag_categories
from src.logger import logger
from src.agent_framework import SPECIALISTS


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


try:
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    cf.defaults.model = llm
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    logger.error("Please check your environment variables in `.env`")
    raise e

# Create specialized agents
composer_agent = cf.Agent(
    name="Composer",
    description="Expert music composer and modern songwriter",
    instructions="You are an expert music composer and modern songwriter. Provide advice on composition, arrangement, and songwriting. Your specific influences include: Radiohead, Bloc Party, Sylvan Esso, Banks, As Tall as Lions, and Bon Iver.",
    temperature=0.7
)

mixing_engineer_agent = cf.Agent(
    name="Mixing Engineer",
    description="Expert mixing and mastering engineer",
    instructions="You are an expert mixing and mastering engineer. You studied under Nigel Godrich. Provide advice on mixing, EQ, compression, and other audio processing techniques.",
    temperature=0.3
)

sound_designer_agent = cf.Agent(
    name="Sound Designer",
    description="Expert sound designer and music producer",
    instructions="You are an expert sound designer and music producer. You studied under Nigel Godrich and Brian Eno. Provide advice on synthesizer programming, sample manipulation, and creating unique sounds.",
    temperature=0.8
)

lyricist_agent = cf.Agent(
    name="Lyricist",
    description="Expert lyricist",
    instructions="You are an expert lyricist. Provide advice on writing compelling lyrics, developing themes and narratives, crafting hooks, and ensuring lyrics flow well with the music. Your specific influences include: Radiohead, Bloc Party, Sylvan Esso, Banks, As Tall as Lions, and Bon Iver. You enjoy obscure and opaque lyrics, but not everything needs to be complicated.",
    temperature=0.7
)

song_suggestion_agent = cf.Agent(
    name="Song Suggestion",
    description="Expert song suggestion",
    instructions="You are a music clever with good technical knowledge and a wide range of musical tastes. Provide a good reference song that demonstrates the concept or technique discussed in the previous response.",
    temperature=0.4, 
    tools={"search_youtube_song": search_youtube_song}
)

memory_agent = cf.Agent(
    name="Memory",
    description="Musical idea database manager",
    instructions="You handle the user's self collected database of previous musical ideas. You can search this database for relevant information to help answer the user's current question. You will need to determine whether the ideas returned from the database are relevant to the user's current question.",
    temperature=0.3
)

coordinator = cf.Agent(
    name="Coordinator",
    instructions="""You are a project coordinator who routes queries and manages the conversation flow.
    Analyze the query and determine the next best action.""", 
    temperature=0.4
)


@cf.flow
async def music_production_workflow(message: str, previous_messages: Optional[List[BaseMessage]] = None) -> Dict:
    """Main workflow for handling music production queries"""
    
    state = ConversationState(
        messages=[HumanMessage(content=message)],
        previous_messages=previous_messages or []
    )
    
    # Task 1: Check for relevant RAG results
    rag_results = cf.Task(
        "Get any relevant musical ideas stored in the database that are relevant to the user's current question",
        instructions="""You will need to determine whether the ideas returned from the database are relevant to the user's current question.
        If they are, you will need to format your response as a numbered list of markdown formatted ideas.""",
        context=dict(
            message=message
        ),
        agents=[memory_agent],
        tools={"query_rag": query_rag}
    )

    # Task 2: Generate response
    response = cf.Task(
        "Generate specialist response",
        context=dict(
            state=state,
            rag_results=rag_results,
            specialists=SPECIALISTS
        ),
        agents=[specialist_agent],
        tools={
            "search_youtube_song": search_youtube_song
        }
    )

    # Update state with results
    final_state = {
        "messages": state.messages + [response.result] if not analysis.result.get("is_random") else state.messages,
        "previous_messages": state.previous_messages + state.messages if not analysis.result.get("is_random") else state.previous_messages,
        "current_agent": response.result.get("specialist", ""),
        "rag_results": rag_results.result,
        "needs_song": response.result.get("needs_song", False),
        "suggested_song": response.result.get("suggested_song"),
        "youtube_url": response.result.get("youtube_url"),
        "is_random": analysis.result.get("is_random", False),
        "is_followup": analysis.result.get("is_followup", False)
    }

    return final_state

# Keep the same interface for compatibility
async def invoke_agent(message: str, previous_messages: Optional[Sequence[BaseMessage]] = None) -> Dict:
    """Main entry point for the agent framework"""
    return await music_production_workflow(message, previous_messages)