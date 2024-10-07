from typing import AsyncGenerator, List

from src.modules.bot.chains import full_chain


async def chatbot(message: str, history: List) -> AsyncGenerator:
    try:
        chunks = ""
        async for response in full_chain().astream(
            input={
                "message": message,
                "history": history,
            }
        ):
            chunks += response
            yield chunks
    except Exception as e:  # pylint: disable=broad-except
        yield {"message": f"An error occurred: {e}"}
