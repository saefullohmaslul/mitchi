from typing import AsyncGenerator, List

import gradio as gr
from dotenv import load_dotenv

from src.modules.bot import chatbot

load_dotenv()

# from arxiv import Client, Search, SortCriterion


async def predict(message: str, history: List) -> AsyncGenerator:
    """Predict the next message in the conversation.

    Args:
        message (_type_): message to predict
        history (_type_): conversation history

    Yields:
        _type_: predicted message
    """
    # client = Client(page_size=10)
    # search = Search(query=message, max_results=10, sort_by=SortCriterion.SubmittedDate)

    # research = []
    # for r in client.results(search):
    #     research.append(
    #         {
    #             "title": r.title,
    #             "summary": r.summary,
    #             "authors": [{"name": a.name} for a in r.authors],
    #             "doi": r.doi,
    #             "url": r.entry_id,
    #         }
    #     )

    # print(research)

    history_message = []
    for human, assistant in history:
        history_message.append({"role": "user", "content": human})
        history_message.append({"role": "ai", "content": assistant})

    async for response in chatbot(message=message, history=history_message):
        yield response


if __name__ == "__main__":
    interface = gr.ChatInterface(predict)
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
