import os
from typing import Any, Dict, List, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import Graph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from src.rag import get_random_by_category, get_ideas_db, search_youtube_song, _rag_categories
from src.logger import logger



SPECIALISTS = {
    "composer": {
        "temperature": 0.7,
        "system_message": SystemMessage(content="You are an expert music composer. Provide advice on composition, arrangement, and songwriting.")
    },
    "mixing_engineer": {
        "temperature": 0.3,
        "system_message": SystemMessage(content="You are an expert mixing engineer. Provide advice on mixing, EQ, compression, and other audio processing techniques.")
    },
    "sound_designer": {
        "temperature": 0.8,
        "system_message": SystemMessage(content="You are an expert sound designer. Provide advice on synthesizer programming, sample manipulation, and creating unique sounds.")
    },
    "lyricist": {
        "temperature": 0.7,
        "system_message": SystemMessage(content="You are an expert lyricist and songwriter. Provide advice on writing compelling lyrics, developing themes and narratives, crafting hooks, and ensuring lyrics flow well with the music.")
    },
    "project_manager": {
        "temperature": 0.4,
        "system_message": SystemMessage(content="You are a music project manager. Coordinate the efforts and provide overall guidance on the music production process.")
    }
}

# Type definitions for our state
class AgentState(TypedDict):
    """State container for the agent workflow.
    
    Attributes:
        messages: Sequence of conversation messages
        previous_messages: Sequence of previous conversation messages
        current_agent: Currently selected specialist agent
        rag_results: Retrieved context from vector store
        needs_song: Flag indicating if a song example is needed
        suggested_song: Recommended song for reference
        youtube_url: YouTube URL for the suggested song
        is_random: Flag indicating if user requested a random idea
        is_followup: Flag indicating if this is a followup question
    """
    messages: Sequence[BaseMessage]
    previous_messages: Sequence[BaseMessage]
    current_agent: str
    rag_results: List[Any]
    needs_song: bool
    suggested_song: str | None
    youtube_url: str | None
    is_random: bool
    is_followup: bool

# Initialize our LLM
try:
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    logger.error("Please check your environment variables in `.env`")
    raise e

def is_random_request(state: Dict) -> Dict:
    """Check if the user is requesting a random idea."""
    last_message = state["messages"][-1].content.lower()
    is_random = "random idea" in last_message
    return {**state, "is_random": is_random}


async def get_random_idea_rag(state: Dict) -> Dict:
    """Handle requests for random ideas."""
    last_message = state["messages"][-1].content.lower()
    category = None
    
    categories_in_message = [category for category in _rag_categories if category.lower() in last_message]
    if categories_in_message:
        category = categories_in_message
    
    result = await get_random_by_category(category)
    if result:
        return {**state, "rag_results": [result]}
    return state


async def query_vectorstore(state: Dict) -> Dict:
    """Query the vectorstore and add results to state."""
    last_message = state["messages"][-1].content
    results = await get_ideas_db(last_message, k=3)
    
    # Return updated state with all existing fields
    return {**state, "rag_results": results}


def filter_relevant_ideas(state: Dict) -> Dict:
    """Filter RAG results to only those relevant to the user's query.
    
    Args:
        state: Current agent state containing messages and RAG results
        
    Returns:
        Updated state with filtered RAG results
    """
    filter_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""You are a filter that determines which ideas are relevant to the user's query.
                    Analyze each idea and return a JSON list of indices for relevant ideas only.
                    Example: [0, 2] if only the first and third ideas are relevant."""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "RAG Results to filter: {rag_results}")
    ])

    response = llm.invoke(
        filter_prompt.format_messages(
            messages=state["messages"],
            rag_results=state["rag_results"]
        )
    )
    try:
        relevant_indices = JsonOutputParser().parse(response.content)
        logger.info(f"Relevant RAG indices: {relevant_indices}")
        filtered_results = [state["rag_results"][i] for i in relevant_indices]
        return {**state, "rag_results": filtered_results}
    except Exception as e:
        logger.error(f"Error parsing filter response: {e}")
        return {**state, "rag_results": state["rag_results"]}


def select_specialist(state: Dict) -> Dict:
    """Select the most appropriate specialist for the query."""
    specialist_selector_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""You are a project coordinator. Your job is to route queries to the appropriate specialist.
                    Choose the most appropriate specialist:
                    - composer: For composition, arrangement, and music theory
                    - mixing_engineer: For audio processing, mixing, mastering and sound balancing
                    - sound_designer: For audio synthesis, sound creation, and sonic textures
                    - lyricist: For writing lyrics, themes, hooks and song narratives
                    - project_manager: For project coordination or unclear queries
                    
                    Respond only with the role key, nothing else."""), 
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = llm.invoke(
        specialist_selector_prompt.format_messages(messages=state["previous_messages"] + state["messages"])
    )
    
    # Return updated state with all existing fields
    return {**state, "current_agent": response.content.strip()}


def specialist_response(state: Dict) -> Dict:
    """Generate a response from the selected specialist agent.
    
    Args:
        state: Current agent state containing messages and specialist selection
        
    Returns:
        Updated state with specialist's response added to messages
    """
    specialist_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="previous_messages"),
        ("user", "Query: {query}")
    ])

    response = llm.invoke(
        specialist_prompt.format(
            query=state["messages"][-1].content,
            agent_scratchpad=[SPECIALISTS[state["current_agent"]]["system_message"]],
            previous_messages=state["previous_messages"]
        )
    )
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


def needs_song_suggestion(state: Dict) -> bool:
    """Determine if a song suggestion is needed based on the specialist's response."""
    song_check_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""Analyze the conversation and determine whether it would be useful to provide a reference track suggestion to demonstrate the concept or technique discussed in the previous response.
                    Respond ONLY with "YES" or "NO"."""), 
        AIMessage("Previous response: {specialist_response}")
    ])

    response = llm.invoke(
        song_check_prompt.format(specialist_response=state["messages"][-1].content),
        temperature=0.3
    )
    logger.info(f"Needs song suggestion: {response.content.strip()}")
    
    try:
        needs_song = "YES" in response.content.strip().upper()
    except Exception as e:
        logger.error(f"Error parsing song suggestion response: {e}")
        needs_song = False
    
    # TODO: This is a hack to get the song suggestion to run every time
    return {**state, "needs_song": needs_song}

async def get_song_suggestion(state: Dict) -> Dict:
    """Get song suggestion and YouTube URL when needed."""
    song_suggester_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""You are an experienced music producer. 
                    Suggest ONE reference song that demonstrates the concept or technique discussed in the previous response.
                    Format your response as "SONG_SUGGESTION: Artist - Song Title"."""), 
        ("user", "Previous response: {specialist_response}")
    ])

    response = llm.invoke(
        song_suggester_prompt.format(specialist_response=state["messages"][-1].content)
    )
    
    song = response.content.split("SONG_SUGGESTION:")[1].strip()
    logger.info(f"Suggesting song: {song}")
    youtube_url = await search_youtube_song(song)
    
    return {**state, "suggested_song": song, "youtube_url": youtube_url}

def is_followup_question(state: Dict) -> Dict:
    """Check if this is a followup question based on message history length."""
    is_followup = len(state["previous_messages"]) > 0
    logger.info(f"Is followup question: {is_followup}")
    return {**state, "is_followup": is_followup}

# Graph construction
def create_agent_graph() -> Graph:
    """Create the workflow graph for the agent framework.
    
    The graph defines the following workflow:
    1. Check if user requested random idea
    2. Either get random idea or select specialist
    3. Query vector store for relevant context
    4. Filter results for relevance
    5. Generate specialist response
    6. Optionally suggest reference song
    
    Returns:
        Compiled workflow graph
    """
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("is_followup", is_followup_question)
    workflow.add_node("is_random_request", is_random_request)
    workflow.add_node("get_random_idea_rag", get_random_idea_rag)
    workflow.add_node("select_specialist", select_specialist)
    workflow.add_node("query_rag", query_vectorstore)
    workflow.add_node("filter_rag", filter_relevant_ideas)
    workflow.add_node("specialist", specialist_response)
    workflow.add_node("needs_song", needs_song_suggestion)
    workflow.add_node("get_song_and_link", get_song_suggestion)
    
    # Connect the nodes
    workflow.add_edge(START, "is_followup")
    workflow.add_edge("get_random_idea_rag", END)
    workflow.add_edge("query_rag", "filter_rag")
    workflow.add_edge("filter_rag", "select_specialist")
    workflow.add_edge("select_specialist", "specialist")
    workflow.add_edge("specialist", "needs_song")
    workflow.add_edge("get_song_and_link", END)

    # Define the conditional edges
    workflow.add_conditional_edges(
        "is_followup",
        lambda state: state["is_followup"],
        {
            True: "select_specialist",  
            False: "is_random_request"  
        }
    )
    
    workflow.add_conditional_edges(
        "is_random_request",
        lambda state: state["is_random"],
        {
            True: "get_random_idea_rag",
            False: "query_rag"
        }
    )
    
    workflow.add_conditional_edges(
        "needs_song",
        lambda state: state["needs_song"],
        {
            True: "get_song_and_link",
            False: END
        }
    )
    
    return workflow.compile()

# Update the invoke_agent function to be async
async def invoke_agent(message: str, previous_messages: Optional[Sequence[BaseMessage]] = None) -> Dict:
    """Main entry point for the agent framework.
    
    Args:
        message: User input message
        previous_messages: Optional list of previous conversation messages
        
    Returns:
        Final state containing agent response and conversation history
    """
    graph = create_agent_graph()

    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        previous_messages=previous_messages,
        current_agent="",
        rag_results=[],
        needs_song=False,
        suggested_song=None,
        youtube_url=None,
        is_random=False,
        is_followup=False
    )
    
    result = await graph.ainvoke(initial_state)

    # dont add previous messages if they are only requests for RAG
    if any(isinstance(msg, AIMessage) for msg in result["messages"]):
        result["previous_messages"] += result["messages"]
    return result
