import autogen
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
