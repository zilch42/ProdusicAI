from importlib import metadata
from nicegui import ui, app
from langchain.schema import AIMessage

from src.rag import convert_timestamp_to_yt, query_rag, _rag_categories
from src.logger import NiceGuiLogElementCallbackHandler, nicegui_handler
from src.agent_framework import invoke_agent

app.add_static_files('/img', 'img')

def create_youtube_embed(url: str, timestamp: str = "") -> str:
    """Create an HTML iframe for a YouTube URL."""
    video_id = url.split('=')[-1]
    return f"""
        <iframe 
            width="560" height="315" 
            src="https://www.youtube.com/embed/{video_id}?si=dTVYtHXBIJeY_EH-{timestamp}" 
            title="YouTube video player" frameborder="0" 
            allow="clipboard-write; encrypted-media; picture-in-picture" 
            referrerpolicy="strict-origin-when-cross-origin" 
            allowfullscreen>
        </iframe>
    """

@ui.page('/')
def main():
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
    
    # Create callback handler with log element
    callback_handler = NiceGuiLogElementCallbackHandler(log_element)
    # llm.callbacks = [callback_handler]
    
    # Connect the log element to both handlers
    nicegui_handler.set_log_element(log_element)
    callback_handler.set_log_element(log_element)
    
    def reset_conversation():
        """Clear the chat history and restore suggested prompts."""
        message_container.clear()
        show_suggested_prompts()  # Show suggested prompts when conversation is reset

    async def send() -> None:
        """Process and send user message to the AI assistant.
        
        Handles:
        - Displaying user message
        - Retrieving and showing relevant RAG results
        - Getting and displaying AI assistant response
        - Managing UI elements (spinners, scroll behavior)
        """
        user_message = text.value
        if not user_message:
            return
        
        text.value = ''
        
        # Add user message
        with message_container:
            ui.chat_message(text=user_message, 
                            name='You', 
                            sent=True, 
                            avatar='img/sprite.png')\
                            .classes('q-pa-md')\
                            .props('text-color="black" bg-color="orange-3"')
            ideas_response = ui.chat_message(name='Relevant Ideas', 
                                             sent=False,
                                             avatar='img/db_icon.png')\
                            .classes('q-pa-md')\
                            .props('text-color="black" bg-color="brown-3"')
            specialist_message = ui.chat_message(name='Selecting Assistant...', 
                                                 sent=False, 
                                                 avatar='img/logo2.png')\
                            .classes('q-pa-md')\
                            .props('text-color="black" bg-color="blue-3"')
            spinner = ui.spinner(type='audio', size='3em')
        
        # Hide suggested prompts when a message is sent
        suggested_prompts_container.clear()
        
        # Get response from agent framework
        result = await invoke_agent(user_message)
        
        # Show RAG results if any
        if result["rag_results"]:
            with ideas_response:
                for doc in result["rag_results"]:
                    idea_text = f"## {doc.metadata['Technique']}\n{doc.metadata['Description']}"
                    if doc.metadata.get('Song'):
                        idea_text += f"\n\n*Reference: {doc.metadata['Song']}*"
                    ui.markdown(idea_text)
                    if doc.metadata.get('Link'):
                        link = doc.metadata.get('Link')
                        if isinstance(link, str) and link.startswith('['):
                            links = eval(link)
                            timestamps = doc.metadata.get('Timestamp', None)
                            timestamps = eval(timestamps) if timestamps else [""]*len(links)
                            for youtube_link, ts in zip(links, timestamps):
                                ts_param = convert_timestamp_to_yt(ts)
                                ui.html(create_youtube_embed(youtube_link, ts_param))
                        else:
                            ts = convert_timestamp_to_yt(doc.metadata.get('Timestamp'))
                            ui.html(create_youtube_embed(link, ts))
                        await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        else:
            message_container.remove(ideas_response)
        
        # Show specialist response 
        with specialist_message as sm:
            specialist_name = result["current_agent"].replace('_', ' ').title()
            sm.props(f'name="{specialist_name}"')
            
            # Get the last AI message from the messages list
            ai_message = next((msg for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)), None)
            if ai_message:
                ui.markdown(ai_message.content)
                
                # If there's a verified YouTube example, show it
                if result.get("youtube_url"):
                    ui.html(create_youtube_embed(result["youtube_url"]))
                    
            await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        
        message_container.remove(spinner)
        await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

    # Suggested prompts container
    suggested_prompts_container = ui.column().classes('w-full max-w-2xl mx-auto my-6 items-center')

    def show_suggested_prompts():
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
        
        def display_prompt_cards(text_list, prefix: str = ""):
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
            ui.label("Get random ideas from database:").classes('q-ma-md').style('font-size: 200%; font-weight: 300')
            display_prompt_cards(_rag_categories, "Give me a random idea for")
            ui.label("Quick start chat prompts:").classes('q-ma-md').style('font-size: 200%; font-weight: 300')
            display_prompt_cards(prompts)

    async def enter_prompt(prompt: str):
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

ui.run(title='Music Production Assistant', host='0.0.0.0', port=8001)
        