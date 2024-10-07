import logging

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict(message, history):
    """Predict the next message in the conversation.

    Args:
        message (_type_): message to predict
        history (_type_): conversation history

    Yields:
        _type_: predicted message
    """
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    # response = client.chat.completions.create(
    # model="gpt-3.5-turbo",
    # messages=history_openai_format,
    # temperature=1.0,
    # stream=True,
    # )

    partial_message = "Halo"
    # for chunk in response:
    # if chunk.choices[0].delta.content is not None:
    # partial_message = partial_message + chunk.choices[0].delta.content
    yield partial_message


if __name__ == "__main__":
    interface = gr.ChatInterface(predict)
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
