import os
from typing import Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import Graph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from src.rag import get_random_by_category, query_rag, search_youtube_song
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
    messages: Sequence[BaseMessage]
    current_agent: str
    rag_results: List[Any]
    needs_song: bool
    suggested_song: str | None
    youtube_url: str | None
    is_random: bool

# Initialize our LLM
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

def is_random_request(state: Dict) -> Dict:
    """Check if the user is requesting a random idea."""
    last_message = state["messages"][-1].content.lower()
    is_random = "random idea" in last_message
    return {**state, "is_random": is_random}


async def get_random_idea_rag(state: Dict) -> Dict:
    """Handle requests for random ideas."""
    last_message = state["messages"][-1].content.lower()
    category = None
    
    if "category:" in last_message:
        category = last_message.split("category:")[1].strip()
    
    result = await get_random_by_category(category)
    if result:
        return {**state, "rag_results": [result]}
    return state

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
        specialist_selector_prompt.format_messages(messages=state["messages"])
    )
    
    # Return updated state with all existing fields
    return {**state, "current_agent": response.content.strip()}


async def query_vectorstore(state: Dict) -> Dict:
    """Query the vectorstore and add results to state."""
    last_message = state["messages"][-1].content
    results = await query_rag(last_message, k=3)
    
    # Return updated state with all existing fields
    return {**state, "rag_results": results}


def filter_relevant_ideas(state: Dict) -> Dict:
    """Filter RAG results to only those relevant to the query."""
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



def specialist_response(state: Dict) -> Dict:
    """Generate specialist response without song suggestions."""
    specialist_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", """Query: {query}
                    
                    Provide expert advice to the user's query. 
                    Focus on explaining techniques and concepts clearly.""")
    ])

    response = llm.invoke(
        specialist_prompt.format(
            query=state["messages"][-1].content,
            agent_scratchpad=[SPECIALISTS[state["current_agent"]]["system_message"]]
        )
    )
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

def needs_song_suggestion(state: Dict) -> bool:
    """Determine if a song suggestion is needed based on the specialist's response."""
    song_check_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""Analyze the previous response:
                    - If it's a list of options/suggestions, respond with "NO"
                    - If it explains a specific technique/concept that could benefit from a song example, respond with "YES"
                    Respond ONLY with "YES" or "NO"."""), 
        ("user", "Previous response: {previous_response}")
    ])

    response = llm.invoke(
        song_check_prompt.format(previous_response=state["messages"][-1].content)
    )
    logger.info(f"Needs song suggestion: {response.content.strip().upper()}")
    return {**state, "needs_song": "YES" in response.content.strip().upper()}

async def get_song_suggestion(state: Dict) -> Dict:
    """Get song suggestion and YouTube URL when needed."""
    song_suggester_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""You are a music recommendation specialist. 
                    Suggest ONE perfect song that demonstrates the concept or technique discussed in the previous response.
                    Format your response as "SONG_SUGGESTION: Artist - Song Title"."""), 
        ("user", "Previous response: {previous_response}")
    ])

    response = llm.invoke(
        song_suggester_prompt.format(previous_response=state["messages"][-1].content)
    )
    
    song = response.content.split("SONG_SUGGESTION:")[1].strip()
    logger.info(f"Suggesting song: {song}")
    youtube_url = await search_youtube_song(song)
    
    return {**state, "suggested_song": song, "youtube_url": youtube_url}

# Graph construction
def create_agent_graph() -> Graph:
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("is_random_request", is_random_request)
    workflow.add_node("get_random_idea_rag", get_random_idea_rag)
    workflow.add_node("select_specialist", select_specialist)
    workflow.add_node("query_rag", query_vectorstore)
    workflow.add_node("filter_rag", filter_relevant_ideas)
    workflow.add_node("specialist", specialist_response)
    workflow.add_node("needs_song", needs_song_suggestion)
    workflow.add_node("get_song_and_link", get_song_suggestion)
    
    # Define the conditional edges
    workflow.add_conditional_edges(
        "is_random_request",
        lambda state: state["is_random"],
        {
            True: "get_random_idea_rag",
            False: "select_specialist"
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
    
    # Connect the nodes
    workflow.add_edge(START, "is_random_request")
    workflow.add_edge("get_random_idea_rag", END)
    workflow.add_edge("select_specialist", "query_rag")
    workflow.add_edge("query_rag", "filter_rag")
    workflow.add_edge("filter_rag", "specialist")
    workflow.add_edge("specialist", "needs_song")
    workflow.add_edge("get_song_and_link", END)

    return workflow.compile()

# Update the invoke_agent function to be async
async def invoke_agent(message: str) -> Dict:
    """Main entry point for the agent framework."""
    graph = create_agent_graph()
    
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        current_agent="",
        rag_results=[],
        needs_song=False,
        suggested_song=None,
        youtube_url=None,
        is_random=False
    )
    
    result = await graph.ainvoke(initial_state)  # Use ainvoke instead of invoke
    return result
