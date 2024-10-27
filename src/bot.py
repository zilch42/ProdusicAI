from logger import logger, log_function, log_rag_query
from agents import user_proxy, composer_agent, mixing_engineer, sound_designer, lyrics_agent, project_manager
from rag import initialize_rag

@log_rag_query
@log_function
def query_rag(query, k=3):
    """
    Query the RAG system to find similar documents based on semantic search.

    Args:
        query (str): The search query text
        k (int, optional): Number of results to return. Defaults to 3.

    Returns:
        List[Document]: List of Document objects, where each Document contains:
            - page_content (str): Text content in format "Category: X\nTechnique: Y\nDescription: Z"
            - metadata (dict): Dictionary containing metadata fields like:
                - Category (str): The idea category
                - Technique (str): The production technique
                - Description (str): Detailed description
                - Song (str, optional): Related song reference
                - Link (str, optional): YouTube link if song exists
    """
    results = vectorstore.similarity_search(query, k=k)
    return results  

@log_function
def music_production_chat():
    logger.info("Starting music production chat session")
    user_proxy.initiate_chat(
        project_manager,
        message="Hello, I need help with my music production project.",
    )

    while True:
        print("DEBUG: Loop iteration", flush=True)  # Add this line
        logger.debug("Starting new iteration of chat loop")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            logger.info("Ending chat session")
            break

        # Concatenate relevant ideas into single string
        results = query_rag(user_input)
        ideas_text = "\n".join([f"- {doc.page_content}" for doc in results])
        print(f"Retrieved relevant ideas:\n{ideas_text}")
        
        # Prepare the message with user input and concatenated ideas
        message = f"User input: {user_input}\n\nRelevant previous ideas:\n{ideas_text}"

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
        print(f"Project manager routed query to: {routing_response}")
        
        # Map the response to the appropriate agent
        recipient = {
            "composer agent": composer_agent,
            "mixing engineer": mixing_engineer,
            "sound designer": sound_designer,
            "lyrics agent": lyrics_agent,
            "project manager": project_manager
        }.get(routing_response.lower().strip(), project_manager)

        print(f"Sending message to {recipient.name}")
        user_proxy.send(message, recipient)


if __name__ == "__main__":
    vectorstore = initialize_rag()
    music_production_chat()
