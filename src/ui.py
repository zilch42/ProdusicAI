from nicegui import ui
from rag import query_rag
from agents import get_specialist, llm, SPECIALISTS
from langchain.schema import HumanMessage
import asyncio

class MusicProductionUI:
    def __init__(self):
        self.chat_messages = []
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
                    .classes('w-full')
                ui.button('Send', on_click=self.handle_message) \
                    .classes('bg-blue-600 text-white')

    async def handle_message(self):
        user_message = self.input.value
        if not user_message:
            return

        # Clear input
        self.input.value = ''

        # Add user message to UI
        with self.messages_container:
            ui.label(f"You: {user_message}").classes('text-right')

        # Get relevant ideas from RAG
        results = query_rag(user_message)
        ideas_text = "\n".join([f"- {doc.page_content}" for doc in results])

        # Show retrieved ideas
        with self.messages_container:
            with ui.card().classes('bg-gray-100'):
                ui.label('Relevant Ideas:')
                for doc in results:
                    ui.markdown(f"- {doc.page_content}")
                    if doc.metadata.get('Song'):
                        ui.link(
                            f"Reference: {doc.metadata['Song']}", 
                            doc.metadata.get('Link', '#')
                        ).classes('text-sm text-blue-600')

        # Determine specialist
        specialist = get_specialist(user_message)
        
        # Create conversation chain
        messages = [
            SPECIALISTS[specialist]["system_message"],
            HumanMessage(content=f"""User query: {user_message}

Relevant context and ideas:
{ideas_text}

Please provide expert advice based on your specialization.""")
        ]

        # Get response from specialist
        response = await asyncio.to_thread(
            llm.invoke,
            messages,
            temperature=SPECIALISTS[specialist]["temperature"]
        )

        # Add specialist response to UI
        with self.messages_container:
            with ui.card().classes('bg-blue-100'):
                ui.label(f"{specialist.replace('_', ' ').title()}:").classes('font-bold')
                ui.markdown(response.content)

