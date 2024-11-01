from importlib import metadata
from nicegui import ui, app
from langchain.schema import AIMessage

from src.agents import AgentManager
from src.rag import convert_timestamp_to_yt, query_rag
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
    
    # Initialize agent manager with callback handler
    agent_manager = AgentManager(callback_handler=callback_handler)
    
    def reset_conversation():
        message_container.clear()
        agent_manager.reset()  

    async def send() -> None:
        user_message = text.value
        if not user_message:
            return
        
        text.value = ''
        
        # Add user message
        with message_container:
            ui.chat_message(text=user_message, name='You', sent=True, avatar='img/sprite.png').classes('q-pa-md')\
                .props('text-color="black" bg-color="orange-3"')
            ideas_response = ui.chat_message(name='Relevant Ideas', sent=False).classes('q-pa-md')\
                .props('text-color="black" bg-color="brown-3"')
            specialist_message = ui.chat_message(name='Assistant', sent=False, avatar='img/logo2.png').classes('q-pa-md')\
                .props('text-color="black" bg-color="primary"')
            spinner = ui.spinner(type='audio', size='3em')
        
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
        specialist_message.clear()
        with specialist_message:
            specialist_name = result["current_agent"].replace('_', ' ').title()
            ui.label(f"{specialist_name}:").classes('font-bold')
            
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

    # Footer with input
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            text = ui.input(placeholder='Ask me anything about music production...') \
                .props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)
            ui.button(icon='delete_forever', on_click=reset_conversation).props('flat').tooltip('clear conversation')

ui.run(title='Music Production Assistant', host='0.0.0.0', port=8001)
        