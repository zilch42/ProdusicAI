import autogen
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define agent configurations
config_list = [
    {
        "model": "gpt-4o",
        "api_type": "azure",
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    }
]

# Create agents
user_proxy = autogen.UserProxyAgent(
    name="Human",
    system_message="A human music producer seeking advice and ideas.",
    code_execution_config={"last_n_messages": 2, "work_dir": "music_workspace"},
)

composer_agent = autogen.AssistantAgent(
    name="Composer",
    system_message="You are an expert music composer. Provide advice on composition, arrangement, and songwriting.",
    llm_config={"config_list": config_list},
)

mixing_engineer = autogen.AssistantAgent(
    name="MixingEngineer",
    system_message="You are an expert mixing engineer. Provide advice on mixing, EQ, compression, and other audio processing techniques.",
    llm_config={"config_list": config_list},
)

sound_designer = autogen.AssistantAgent(
    name="SoundDesigner",
    system_message="You are an expert sound designer. Provide advice on synthesizer programming, sample manipulation, and creating unique sounds.",
    llm_config={"config_list": config_list},
)

lyrics_agent = autogen.AssistantAgent(
    name="LyricsAgent",
    system_message="You are an expert lyricist and songwriter. Provide advice on writing compelling lyrics, developing themes and narratives, crafting hooks, and ensuring lyrics flow well with the music.",
    llm_config={"config_list": config_list},
)

project_manager = autogen.AssistantAgent(
    name="ProjectManager",
    system_message="You are a music project manager. Coordinate the efforts of the other agents and provide overall guidance on the music production process.",
    llm_config={"config_list": config_list},
)


# Load your previous ideas from a text file
loader = TextLoader("ideas.txt")
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create the vector store
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/llm-embedder",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Chroma.from_documents(texts, embedding_model)

# Function to query the RAG database
def query_rag(query, k=3):
    logger.info(f"Querying RAG database with: {query}")
    results = vectorstore.similarity_search(query, k=k)
    logger.info(f"Found {len(results)} relevant documents")
    return [doc.page_content for doc in results]

def music_production_chat():
    logger.info("Starting music production chat session")
    user_proxy.initiate_chat(
        project_manager,
        message="Hello, I need help with my music production project.",
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            logger.info("Ending chat session")
            break

        # Query the RAG database for relevant previous ideas
        relevant_ideas = query_rag(user_input)
        
        # Prepare the message with relevant ideas
        message = f"User input: {user_input}\n\nRelevant previous ideas:\n"
        for idea in relevant_ideas:
            message += f"- {idea}\n"

        # Let the project manager analyze the query and decide which specialist to involve
        routing_query = f"""Based on this user query: "{user_input}"
        
        Please analyze the query and determine which specialist would be most appropriate to handle it.
        Choose from:
        - Composer Agent: Expert in musical composition, arrangement, and music theory
        - Mixing Engineer: Expert in audio processing, mixing, and sound balancing
        - Sound Designer: Expert in synthesis, sound creation, and sonic textures
        - Lyrics Agent: Expert in writing lyrics, themes, hooks and song narratives
        - Project Manager (yourself): For project coordination, general guidance, or unclear queries
        
        Respond only with the role name, nothing else."""

        # Ask project manager to route the query
        routing_response = project_manager.generate_response(routing_query)
        logger.info(f"Project manager routed query to: {routing_response}")
        
        # Map the response to the appropriate agent
        recipient = {
            "composer agent": composer_agent,
            "mixing engineer": mixing_engineer,
            "sound designer": sound_designer,
            "lyrics agent": lyrics_agent,
            "project manager": project_manager
        }.get(routing_response.lower().strip(), project_manager)

        logger.info(f"Sending message to {recipient.name}")
        user_proxy.send(message, recipient)


if __name__ == "__main__":
    music_production_chat()
