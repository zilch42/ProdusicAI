import asyncio
import os
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

from logger import logger
from rag import query_rag

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

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

def filter_relevant_ideas(query, results):
    """Filter RAG results to only those relevant to the query.
    
    Args:
        query (str): The user's query
        results (list): List of RAG results to filter
        
    Returns:
        list: Filtered list containing only relevant results
    """
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    
    relevance_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a music production assistant. Your job is to determine which music production ideas are relevant to the query."),
        HumanMessage(content=f"""Query: "{query}"

        For each of these ideas, respond only with 'yes' or 'no' based on whether it is relevant to the query:

        {[doc.page_content for doc in results]}
        
        {format_instructions}""")
    ])
    
    relevance_response = llm.invoke(relevance_prompt.messages)
    logger.info(f"LLM relevance response: {relevance_response.content}")
    
    relevant_flags = [r.strip().lower() == 'yes' for r in output_parser.parse(relevance_response.content)]
    
    return [doc for doc, is_relevant in zip(results, relevant_flags) if is_relevant]


def get_specialist(query):
    """Determine which specialist should handle the query.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: Name of the chosen specialist
    """
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



class AgentManager:
    def __init__(self):
        self.current_specialist = None
        self.message_chain = []

    async def get_rag_results(self, user_message):
        """Get relevant results from the RAG system."""
        # Query the RAG system
        raw_results = await query_rag(user_message)
        
        # Filter results for relevance
        if raw_results:
            return filter_relevant_ideas(user_message, raw_results)
        return []

    async def get_specialist_response(self, user_message, ideas):
        IDEAS_BLURB = """
        The user has a database of previous ideas that they have collected from listening to other songs. 
        If they appear relevant to the query, please include them in your advice.
        You may also include other advice based on your specialization.

        Relevant context and ideas:
        """

        # Add ideas context if available
        if len(ideas) > 0:
            ideas_text = "\n".join([f"- {doc.metadata['Technique']}: {doc.metadata['Description']}" for doc in ideas])
            user_message += IDEAS_BLURB + ideas_text

        # Determine specialist - only if it's new or changed
        new_specialist = get_specialist(user_message)
        if self.current_specialist != new_specialist:
            # Reset chain if specialist changes
            self.message_chain = []
            self.current_specialist = new_specialist
        
        # Add user message to chain
        self.message_chain.append(
            HumanMessage(content=user_message)
        )

        # Create conversation chain with full history
        messages = [
            SPECIALISTS[new_specialist]["system_message"],
            *self.message_chain
        ]

        # Get streaming response from specialist
        response_chunks = []
        async for chunk in llm.astream(
            messages,
            temperature=SPECIALISTS[new_specialist]["temperature"]
        ):
            response_chunks.append(chunk.content)
            # Include specialist info in the yielded data
            yield {
                'content': chunk.content,
                'specialist': new_specialist
            }
        
        full_response = ''.join(response_chunks)
        self.message_chain.append(
            SystemMessage(content=full_response)
        )
