import gradio as gr
import requests
import json
import argparse
from loguru import logger


parser = argparse.ArgumentParser()

parser.add_argument("--endpoint", type=str, default="test_llm", help="URL of the FastAPI endpoint")
parser.add_argument("--use_follow_ups", type=str, default="yes", help="Use follow-ups for the LLM")
parser.add_argument("--port", type=int, default=7860, help="Port where the FastAPI server is running")

args = parser.parse_args()

def query_chatbot(message):
    # FastAPI endpoint URL
    url = f"http://localhost:8000/{args.endpoint}"  # Adjust port if needed
    
    # Make request to FastAPI endpoint
    if args.endpoint == "test_deterministic":
        # For test_llm endpoint, we need to pass the query as a parameter
        response = requests.get(url, params={"query": message, 
                                             "use_follow_ups": args.use_follow_ups})
    else:
        response = requests.get(url, params={"query": message})
    
    if response.status_code == 200:
        result = response.json()
        
        # Format search results for display
        search_results_formatted = json.dumps(result["search_results"], indent=2)
        
        # Return both the answer and search results
        return result["answer"], search_results_formatted
    else:
        return f"Error: {response.status_code}", "No search results available"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Demo")
    
    with gr.Row():
        with gr.Column():
            # Input components
            message_input = gr.Textbox(
                label="Your question",
                placeholder="Type your question here...",
                lines=2
            )
            submit_button = gr.Button("Submit")
        
        with gr.Column():
            # Output components
            answer_output = gr.Markdown(
                label="Answer",
            )
            search_results_output = gr.JSON(
                label="Search Results"
            )
    
    # Handle submission
    submit_button.click(
        fn=query_chatbot,
        inputs=[message_input],
        outputs=[answer_output, search_results_output]
    )
    
    # Also allow submission with Enter key
    message_input.submit(
        fn=query_chatbot,
        inputs=[message_input],
        outputs=[answer_output, search_results_output]
    )

if __name__ == "__main__":
    logger.info(f"Starting Gradio app using the following endpoint: {args.endpoint}")
    demo.launch(share=False, server_port=args.port)  # set share=False if you don't want a public URL