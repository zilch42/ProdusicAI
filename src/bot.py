import autogen
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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

    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def music_production_chat():
    user_proxy.initiate_chat(
        project_manager,
        message="Hello, I need help with my music production project.",
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Query the RAG database for relevant previous ideas
        relevant_ideas = query_rag(user_input)

        # Prepare the message with relevant ideas
        message = f"User input: {user_input}\n\nRelevant previous ideas:\n"
        for idea in relevant_ideas:
            message += f"- {idea}\n"

        # Determine which agent should handle the query
        if "composition" in user_input.lower() or "arrangement" in user_input.lower():
            recipient = composer_agent
        elif "mixing" in user_input.lower() or "eq" in user_input.lower():
            recipient = mixing_engineer
        elif "sound design" in user_input.lower() or "synth" in user_input.lower():
            recipient = sound_designer
        else:
            recipient = project_manager

        # Send the message to the appropriate agent
        user_proxy.send(message, recipient)


if __name__ == "__main__":
    music_production_chat()
