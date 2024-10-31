import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from typing import Optional
from pydantic import BaseModel

from logger import logger
from rag import query_rag, search_youtube_song

load_dotenv()

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

class SongVerification(BaseModel):
    """Schema for song verification results"""
    is_verified: bool
    song_info: Optional[str] = None
    explanation: Optional[str] = None

def verify_and_find_song(song_query: str, technique_context: str) -> SongVerification:
    """
    Verify if a song is a good example for a technique and find its YouTube link.
    
    Args:
        song_query: The song to verify
        technique_context: The technique or concept being demonstrated
        
    Returns:
        SongVerification object containing verification results
    """
    youtube_url = search_youtube_song(song_query)
    return SongVerification(
        is_verified=True if youtube_url else False,
        song_info=song_query,
        explanation=f"Found reference: {youtube_url}" if youtube_url else "Could not find song reference"
    )

def get_specialist(query: str, llm: AzureChatOpenAI) -> str:
    """Determine which specialist should handle the query."""
    routing_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a project coordinator. Your job is to route queries to the appropriate specialist."),
        HumanMessage(content=f"""Based on this query: "{query}"
        
        Choose the most appropriate specialist:
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
    def __init__(self, callback_handler=None):
        self.current_specialist = None
        self.message_chain = []
        self.llm = AzureChatOpenAI(
            openai_api_version="2023-07-01-preview",
            azure_deployment="gpt-4",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            callbacks=[callback_handler] if callback_handler else []
        )

    async def get_rag_results(self, user_message: str):
        """Get relevant results from the RAG system."""
        return await query_rag(user_message)

    async def fact_check_response_songs(self, full_response: str) -> tuple[str, str]:
        """Verify song references using function calling."""
        song_verification_tool = StructuredTool.from_function(
            func=verify_and_find_song,
            name="verify_song",
            description="Verify if a song is a good example for a technique and find its YouTube link",
            return_direct=True
        )

        fact_checker = initialize_agent(
            tools=[song_verification_tool],
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs={
                "system_message": SystemMessage(content="""You are a music expert fact checker. 
                Your job is to verify if suggested reference songs accurately demonstrate musical techniques.
                When you find a valid song reference, use the verify_song tool to get its YouTube link.""")
            }
        )

        fact_check_prompt = f"""
        Review this music advice and verify any song references:
        
        {full_response}
        
        If there's a specific song mentioned as an example:
        1. Verify if it's a good demonstration of the technique
        2. If it is, use the verify_song tool to get its reference
        3. If it's not a good example, explain why
        
        If no specific songs are mentioned or if it's a list of general ideas, respond with 'NO_REFERENCE_NEEDED'
        """

        fact_check_response = await fact_checker.ainvoke({"input": fact_check_prompt})
        response_content = fact_check_response["output"]

        if isinstance(response_content, SongVerification):
            if response_content.is_verified:
                full_response += f"\n\nFact Check: {response_content.song_info}\n{response_content.explanation}"
        elif "NO_REFERENCE_NEEDED" not in response_content:
            full_response += f"\n\nFact Check: {response_content}"

        return full_response, response_content

    async def get_specialist_response(self, user_message: str, ideas: list):
        """Get a response from the appropriate specialist."""
        if ideas:
            ideas_text = "\n".join([f"- {doc.metadata['Technique']}: {doc.metadata['Description']}" for doc in ideas])
            user_message += "\n\nRelevant context and ideas:\n" + ideas_text

        user_message += """\n\nWhen giving specific advice about a particular technique or approach, 
        you may include ONE relevant song as a reference example."""

        new_specialist = get_specialist(user_message, self.llm)
        if self.current_specialist != new_specialist:
            self.message_chain = []
            self.current_specialist = new_specialist
        
        self.message_chain.append(HumanMessage(content=user_message))

        messages = [
            SPECIALISTS[new_specialist]["system_message"],
            *self.message_chain
        ]

        response_chunks = []
        async for chunk in self.llm.astream(
            messages,
            temperature=SPECIALISTS[new_specialist]["temperature"]
        ):
            response_chunks.append(chunk.content)
            
        full_response = ''.join(response_chunks)
        full_response, fact_check_response = await self.fact_check_response_songs(full_response)
        
        self.message_chain.append(SystemMessage(content=full_response))
        
        yield {
            'content': f"\n\nFact Check: {fact_check_response}" if "NO_REFERENCE_NEEDED" not in str(fact_check_response) else "",
            'specialist': new_specialist
        }

    def reset(self):
        """Reset the conversation state."""
        self.message_chain = []
        self.current_specialist = None