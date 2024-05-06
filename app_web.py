import os
from typing import Optional, Tuple
import gradio as gr
from threading import Lock


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []

            from my_lib import ask_llm
            # Run chain and append input.
            output = ask_llm(question=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain Demo</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary")  # .style(full_width=False)

    gr.Examples(
        examples=[
            "What is Transformer?",
            "What steps Data Preprocessing consists of?",
            "Whats 2 + 2?",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    state = gr.State()

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug=True)
