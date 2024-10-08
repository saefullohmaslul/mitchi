from typing import AsyncGenerator, List

import gradio as gr
from dotenv import load_dotenv

from src.modules.bot import chatbot

load_dotenv()

greeting_message = "Halo! Saya adalah Mitchi, asisten risetmu. Bagaimana saya bisa membantu Anda hari ini?"


async def predict(message: str, history: List) -> AsyncGenerator:
    """Predict the next message in the conversation.

    Args:
        message (str): message to predict
        history (List): conversation history

    Yields:
        AsyncGenerator: predicted message
    """
    history_message = []
    for human, assistant in history:
        history_message.append({"role": "user", "content": human})
        history_message.append({"role": "ai", "content": assistant})

    async for response in chatbot(message=message, history=history_message):
        yield response


if __name__ == "__main__":
    interface = gr.ChatInterface(
        fn=predict,
        title="Mitchi: Smart Research Discussion Chatbot",
        examples=[
            "Halo, kamu siapa?",
            "Aku mau bikin penelitian tapi bingung topiknya apa",
            "Aku mau cari referensi tentang machine learning",
        ],
    )
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
