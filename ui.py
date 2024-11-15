from ast import literal_eval
from nicegui import ui, app
from langchain.schema import AIMessage
from src.rag import _rag_categories
from src.logger import logger, nicegui_handler
from src.agent_framework import invoke_agent
from src.youtube import create_youtube_embed, convert_timestamp_to_yt
from src.config import get_config

app.add_static_files('/img', 'img')

def init_app() -> bool:
    """Initialize the application."""
    try:
        config = get_config()
        logger.info("Application initialized successfully")
        return True
    except ValueError as e:
        logger.error(f"Failed to initialize application: {e}")
        return False

@ui.page('/')
def main() -> None:
    if not init_app():
        return ui.markdown("# ⚠️ Configuration Error\nPlease check your environment variables and logs.")
    
    """Main application page that sets up the chat interface.
    
    Creates a tabbed interface with two panels:
    - Chat: Main interaction area with the AI assistant
    - Logs: Debug logging information
    
    The chat interface includes suggested prompts and a message input area.
    """
    # Make UI elements expand properly
    ui.query('.q-page').classes('flex')
    ui.query('.nicegui-content').classes('w-full')
    
    # Create tabs
    with ui.tabs().classes('w-full') as tabs:
        chat_tab = ui.tab('Chat')
        logs_tab = ui.tab('Logs')
    
    # Create tab panels
    with ui.tab_panels(tabs, value=chat_tab).classes('w-full max-w-2xl mx-auto flex-grow items-stretch'):
        message_container = ui.tab_panel(chat_tab).classes('items-stretch')
        with ui.tab_panel(logs_tab):
            log_element = ui.log().classes('w-full h-full')
    
    # Connect the log element to both handlers
    nicegui_handler.set_log_element(log_element)
    
    # Initialize previous_messages list at the start of main()
    previous_messages: list = []
    
    def reset_conversation() -> None:
        """Clear the chat history and restore suggested prompts."""
        message_container.clear()
        previous_messages.clear()
        suggested_prompts_container.clear()
        show_suggested_prompts()

    async def send() -> None:
        """Process and send user message to the AI assistant.
        
        Handles:
        - Displaying user message
        - Retrieving and showing relevant RAG results
        - Getting and displaying AI assistant response
        - Managing UI elements (spinners, scroll behavior)
        """
        nonlocal previous_messages 
        user_message: str = text.value
        if not user_message:
            return
        
        text.value = ''
        
        # Add user message to UI
        with message_container:
            ui.chat_message(text=user_message, 
                            name='You', 
                            sent=True, 
                            avatar='img/sprite.png')\
                            .classes('q-pa-md')\
                            .props('text-color="black" bg-color="orange-3"')
            ideas_response = ui.chat_message(name='Fetching relevant ideas...', 
                                             sent=False,
                                             avatar='img/db_icon.png')\
                            .classes('q-pa-md')\
                            .props('text-color="black" bg-color="brown-3"')
            specialist_message = ui.chat_message(name='Selecting assistant...', 
                                                 sent=False, 
                                                 avatar='img/logo2.png')\
                            .classes('q-pa-md')\
                            .props('text-color="black" bg-color="blue-3"')
            spinner = ui.spinner(type='audio', size='3em')

        await ui.run_javascript("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
        
        suggested_prompts_container.clear()

        # Create and store the new human message
        result: dict = await invoke_agent(user_message, previous_messages=previous_messages)
        previous_messages = result["previous_messages"]

        # Show RAG results only if not a followup question
        if result["rag_results"]:
            with ideas_response as ir:
                ir.props(f'name="Relevant ideas"')
                for doc in result["rag_results"]:
                    try:
                        idea_text = f"## {doc.Technique}\n{doc.Description}"
                    except KeyError as e:
                        logger.error(f"Error in RAG doc: {doc}")
                        logger.error(e)
                        idea_text = "Error with RAG doc metadata"
                    if doc.Song:
                        idea_text += f"\n\n*Reference: {doc.Song}*"
                    ui.markdown(idea_text)
                    if doc.Link:
                        link = doc.Link
                        if isinstance(link, str) and link.startswith('['):
                            links = literal_eval(link)
                            timestamps = doc.Timestamp
                            timestamps = literal_eval(timestamps) if timestamps else [""]*len(links)
                            for youtube_link, ts in zip(links, timestamps):
                                ts_param = convert_timestamp_to_yt(ts)
                                ui.html(create_youtube_embed(youtube_link, ts_param))
                        else:
                            ts = convert_timestamp_to_yt(doc.Timestamp)
                            ui.html(create_youtube_embed(link, ts))
            await ui.run_javascript("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
        else:
            message_container.remove(ideas_response)
        
        if result.get("is_random", False):
            show_suggested_prompts("rag")
            message_container.remove(specialist_message)
        else:
            # Show specialist response
            ai_message = next((msg for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)), None)
            if ai_message:
                with specialist_message as sm:
                    specialist_name = result["current_agent"].replace('_', ' ').title()
                    sm.props(f'name="{specialist_name}"')
                    ui.markdown(ai_message.content)
                
                    # If there's a verified YouTube example, show it
                    if result.get("youtube_url"):
                        logger.info(f"Showing YouTube example: {result['youtube_url']}")
                        ui.html(create_youtube_embed(result["youtube_url"]))
                        
                await ui.run_javascript("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
            else:
                message_container.remove(specialist_message)
        
        message_container.remove(spinner)
        await ui.run_javascript("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")

    # Suggested prompts container
    suggested_prompts_container = ui.column().classes('w-full max-w-2xl mx-auto my-6 items-center')

    def show_suggested_prompts(which: str = "all") -> None:
        """Display two sections of clickable prompt suggestions:
        - Random ideas from the RAG database
        - Quick start chat prompts for common questions
        """
        prompts = [
            "How do I mix vocals?",
            "What effects can I put on cymbals?",
            "How can I improve my bassline?",
            "What is sidechain compression?",
            "How do I master a track?", 
            "What chord progressions sound like Radiohead?"
        ]
        
        def display_prompt_cards(text_list: list, prefix: str = "") -> None:
            """Display a list of clickable prompt cards.
            
            Args:
                text_list (list[str]): List of prompt texts
                prefix (str): Prefix to prepend to each prompt text
            """
            # rows of 2
            for i in range(0, len(text_list), 2):
                with ui.row().classes('w-full justify-center'):
                    for text in text_list[i:i+2]:
                        with ui.card().classes('q-pa-md').style('width: 40%') \
                            .on('click', lambda p=text: enter_prompt(f'{prefix} {p}')) \
                            .props('hoverable') \
                            .classes('cursor-pointer hover:bg-blue-100'):
                            ui.label(text).classes('text-center')

        with suggested_prompts_container:
            if which in ["rag", "all"]:
                with ui.row():
                    ui.icon('casino', color='primary').classes('self-center text-5xl')
                    ui.label("Get random ideas from database:").classes('q-ma-md').style('font-size: 200%; font-weight: 300')
                display_prompt_cards(_rag_categories, "Give me a random idea for")
            if which in ["prompt", "all"]:
                with ui.row():
                    ui.icon('live_help', color='primary').classes('self-center text-5xl')
                    ui.label("Quick start chat prompts:").classes('q-ma-md').style('font-size: 200%; font-weight: 300')
                display_prompt_cards(prompts)

    async def enter_prompt(prompt: str) -> None:
        """Handle when a suggested prompt is clicked."""
        text.value = prompt
        suggested_prompts_container.clear()
        await send()

    # Show suggested prompts initially
    show_suggested_prompts()

    # Footer with input
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            text = ui.input(placeholder='Ask me anything about music production...') \
                .props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)
            ui.button(icon='delete_forever', on_click=reset_conversation).props('flat').tooltip('clear conversation')

ui.run(title='ProdusicAI', 
       favicon='img/logo2.png',
       host='0.0.0.0', 
       port=8001, 
       on_air=True)
