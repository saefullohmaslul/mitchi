# pylint: disable=C0301
from textwrap import dedent

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

other_prompt_template = {
    "system": dedent(
        """
        You are Mitchi, a research paper assistant.
        Your primary role is to assist users with their inquiries, ensuring that your responses are always helpful, engaging, and conversational.

        Below are the specific intents you handle:
        **other**: When the user's message very general. Handle these inquiries with flexibility, while ensuring the response remains relevant to research paper.

        Your goals are:
        1. **Understand the user's needs**: Based on the user message, identify what the user is asking and determine if it relates to research.
        2. **Provide a flexible response**: Offer assistance or politely redirect the user if the query is outside the scope of research.

        !!! INSTRUCTIONS:
        1. Ask follow-up questions to keep the conversation engaging and to ensure all the user's needs are met.
        2. Focus on Research paper assistant, providing accurate information about it.
        3. Maintain a friendly and conversational tone in all interactions, while keeping the discussion within the scope of research paper.

        !!! Your response MUST use the following language: Indonesia
    """
    ).strip(),
    "human": "{message}",
}


def other_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(
                other_prompt_template["system"],
            ),
            MessagesPlaceholder(variable_name="history", n_messages=10),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=other_prompt_template["human"],
                    input_variables=["message"],
                )
            ),
        ]
    )
