from nicegui import ui
from agents import AgentManager
from rag import convert_timestamp_to_yt


class MusicProductionUI:
    def __init__(self):
        self.chat_messages = []
        self.agent_manager = AgentManager()
        self.initialize_ui()

    def initialize_ui(self):
        # Header
        with ui.header().classes('bg-blue-600 text-white'):
            ui.label('Music Production Assistant').classes('text-h6')

        # Main content area
        with ui.column().classes('w-full max-w-3xl mx-auto p-4 space-y-4'):
            # Chat messages container
            self.messages_container = ui.column().classes('w-full space-y-2')
            
            # Input area
            with ui.row().classes('w-full'):
                self.input = ui.input(placeholder='Ask me anything about music production...') \
                    .classes('w-full') \
                    .on('keydown.enter', self.handle_message)
                # TODO apply animation on enter
                ui.button('Send', on_click=self.handle_message) \
                    .classes('bg-blue-600 text-white')

    async def handle_message(self):
        user_message = self.input.value
        if not user_message:
            return

        # Clear input
        self.input.value = ''
        spinner = ui.spinner('bars', size='8em')

        # Add user message to UI
        with self.messages_container:
            ui.label(f"You: {user_message}").classes('text-right')

        # Get response from agent manager
        result = await self.agent_manager.process_query(user_message)

        # Show retrieved ideas
        if len(result['rag_results']) > 0:
            with self.messages_container:
                with ui.card().classes('bg-gray-100'):
                    ui.label('Relevant Ideas:')
                    for doc in result['rag_results']:
                        ui.markdown(f"- {doc.page_content}")
                        if doc.metadata.get('Song'):
                            ui.label(f"Reference: {doc.metadata['Song']}").classes('text-sm font-bold')
                            if doc.metadata.get('Link'):
                                ts = convert_timestamp_to_yt(doc.metadata.get('Timestamp', None))
                                ui.html(f"""
                                    <iframe 
                                        width="560"
                                        height="315" 
                                        src="https://www.youtube.com/embed/{doc.metadata['Link'].split('=')[-1]}?si=dTVYtHXBIJeY_EH-{ts}" 
                                        title="YouTube video player" 
                                        frameborder="0" 
                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                                        referrerpolicy="strict-origin-when-cross-origin" 
                                        allowfullscreen>
                                    </iframe>
                                """)

        # Add specialist response to UI
        with self.messages_container:
            with ui.card().classes('bg-blue-100'):
                ui.label(f"{result['specialist'].replace('_', ' ').title()}:").classes('font-bold')
                ui.markdown(result['response'])

        spinner.delete()
        