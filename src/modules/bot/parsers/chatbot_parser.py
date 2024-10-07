from typing import Union

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser


def string_parser(ai_message: AIMessage) -> Union[str, dict]:
    """Parse the refine query output to only take the message.

     Args:
        ai_message (AIMessage): AI message output.

    Returns:
        Union[str, dict]: Parsed refine query as either a string or a dictionary.
    """
    content = ai_message.content

    if isinstance(content, str):
        return StrOutputParser().parse(content).strip()
    else:
        raise ValueError("Expected refine query to be a string")
