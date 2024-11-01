import os
from typing import Annotated, Any, Dict, List, Sequence, TypedDict
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
    suggested_song: str | None
    youtube_url: str | None

# Initialize our LLM
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

def is_random_request(state: Dict) -> bool:
    """Check if the user is requesting a random idea."""
    last_message = state["messages"][-1].content.lower()
    return "random idea" in last_message


def handle_random_request(state: Dict) -> Dict:
    """Handle requests for random ideas."""
    last_message = state["messages"][-1].content.lower()
    category = None
    
    if "category:" in last_message:
        category = last_message.split("category:")[1].strip()
    
    result = get_random_by_category(category)
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
    print("State in select_specialist:", state)  # Debug print
    
    # Return updated state with all existing fields
    return {**state, "current_agent": response.content.strip()}


def query_vectorstore(state: Dict) -> Dict:
    """Query the vectorstore and add results to state."""
    print("State in query_vectorstore:", state)  # Debug print
    last_message = state["messages"][-1].content
    results = query_rag(last_message, top_k=3)
    
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
        filtered_results = [state["rag_results"][i] for i in relevant_indices]
        return {**state, "rag_results": filtered_results}
    except Exception as e:
        logger.error(f"Error parsing filter response: {e}")
        return {**state, "rag_results": state["rag_results"]}



def specialist_response(state: Dict) -> Dict:
    """Generate specialist response and extract any suggested song."""
    specialist_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", """Query: {query}
                    
                    Provide expert advice to the users query. 
                    If you are providing a list of options, do not provide any example songs and if applicable, 
                    but if you are providing more detailed information on a specific technique, 
                    you may suggest ONE specific song that demonstrates the technique or concept you're explaining. 
                    If suggesting a song, format it as: SONG_EXAMPLE: Artist - Song Title
                    """)
    ])

    response = llm.invoke(
        specialist_prompt.format(
            query=state["messages"][-1].content,
            agent_scratchpad=[SPECIALISTS[state["current_agent"]]["system_message"]]
        )
    )
    
    # Extract song if one was suggested
    content = response.content
    suggested_song = None
    if "SONG_EXAMPLE:" in content:
        song_line = content.split("SONG_EXAMPLE:")[1].split("\n")[0].strip()
        suggested_song = song_line
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=content)],
        "suggested_song": suggested_song
    }


def verify_song_example(state: Dict) -> Dict:
    """Verify if the suggested song is a good example."""
    fact_checker_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""You are a music fact checker. Verify if the suggested song actually 
                    demonstrates the technique or concept being discussed. 
                    Return only 'true' or 'false'."""),
        ("user", """Technique/Concept: {query}
                    Suggested Song: {song}
                    Verify if this song is a good example.""")
    ])

    # TODO change this to a conditional edge     
    if not state["suggested_song"]:
        return state
    
    response = llm.invoke(
        fact_checker_prompt.format(
            query=state["messages"][-1].content,
            song=state["suggested_song"]
        )
    )
    
    # TODO change this to a conditional edge or to use search_youtube_song as a tool 
    if response.content.strip().lower() == "true":
        youtube_url = search_youtube_song(state["suggested_song"])
        return {**state, "youtube_url": youtube_url}
    return {**state, "youtube_url": None, "suggested_song": None}

# Graph construction
def create_agent_graph() -> Graph:
    workflow = Graph()
    
    # Define the conditional edges
    workflow.add_conditional_edges(
        START,
        lambda x: is_random_request(x),
        {
            True: "handle_random",
            False: "select_specialist"
        }
    )
    
    # Define the main flow
    workflow.add_node("handle_random", handle_random_request)
    workflow.add_node("select_specialist", select_specialist)
    workflow.add_node("query_rag", query_vectorstore)
    workflow.add_node("filter_rag", filter_relevant_ideas)
    workflow.add_node("specialist", specialist_response)
    workflow.add_node("fact_check", verify_song_example)
    
    # Connect the nodes
    workflow.add_edge("handle_random", END)
    workflow.add_edge("select_specialist", "query_rag")
    workflow.add_edge("query_rag", "filter_rag")
    workflow.add_edge("filter_rag", "specialist")
    workflow.add_edge("specialist", "fact_check")
    workflow.add_edge("fact_check", END)

    return workflow.compile()

# Update the invoke_agent function to be async
async def invoke_agent(message: str) -> Dict:
    """Main entry point for the agent framework."""
    graph = create_agent_graph()
    
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        current_agent="",
        rag_results=[],
        suggested_song=None,
        youtube_url=None
    )
    
    result = await graph.ainvoke(initial_state)  # Use ainvoke instead of invoke
    return result
