from nicegui import ui
from agents import AgentManager
from rag import convert_timestamp_to_yt

@ui.page('/')
def main():
    agent_manager = AgentManager()
    
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
            log = ui.log().classes('w-full h-full')

    async def send() -> None:
        user_message = text.value
        if not user_message:
            return
        
        text.value = ''
        
        # Add user message
        with message_container:
            ui.chat_message(text=user_message, name='You', sent=True)
            ideas_response = ui.chat_message(name='Relevant Ideas', sent=False)
            specialist_message = ui.chat_message(name='Assistant', sent=False)
            spinner = ui.spinner(type='audio', size='3em')
            
        # First get RAG results
        rag_results = await agent_manager.get_rag_results(user_message)
        
        # Show RAG results if any
        if len(rag_results) > 0:
            with ideas_response:
                ui.label('Relevant Ideas:')
                for doc in rag_results:
                    ui.markdown(f"- {doc.page_content}")
                    if doc.metadata.get('Song'):
                        ui.label(f"Reference: {doc.metadata['Song']}").classes('text-sm font-bold')
                        if doc.metadata.get('Link'):
                            ts = convert_timestamp_to_yt(doc.metadata.get('Timestamp', None))
                            ui.html(f"""
                                <iframe 
                                    width="560" height="315" 
                                    src="https://www.youtube.com/embed/{doc.metadata['Link'].split('=')[-1]}?si=dTVYtHXBIJeY_EH-{ts}" 
                                    title="YouTube video player" frameborder="0" 
                                    allow="clipboard-write; encrypted-media; picture-in-picture" 
                                    referrerpolicy="strict-origin-when-cross-origin" 
                                    allowfullscreen>
                                </iframe>
                            """)
        else:
            message_container.remove(ideas_response)
        
        # Get specialist response
        specialist_message.clear()
        with specialist_message:
            specialist_name = None
            content = ui.markdown()  # Create empty markdown element
            
            # Stream the response
            full_response = ""
            async for chunk in agent_manager.get_specialist_response(user_message, rag_results):
                if chunk['content']:
                    if not specialist_name:
                        specialist_name = chunk['specialist'].replace('_', ' ').title()
                        ui.label(f"{specialist_name}:").classes('font-bold')
                    full_response += chunk['content']
                    content.content = full_response
                    await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        
        message_container.remove(spinner)
        await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

    # Footer with input
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            text = ui.input(placeholder='Ask me anything about music production...') \
                .props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)

ui.run(title='Music Production Assistant', host='0.0.0.0', port=8001)
        