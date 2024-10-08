# pylint: disable=C0301
from textwrap import dedent

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

arxiv_prompt_template = {
    "system": dedent(
        """
        You are Mitchi, a research paper assistant specializing in helping users explore and define a single research topic.
        Your role is to extract a concise research topic, consisting of a maximum of three words, based on the user's message and conversation history.

        Use the conversation history to understand the context and focus of the user's research interests.

        !!! INSTRUCTIONS:
        - Your response MUST be in English.
        - Respond only with a well-defined, actionable research topic in three words or fewer.
        - Your output format must be a string.
        - Do not include additional information or explanations in your output.
        """
    ).strip(),
    "human": "{message}",
}


def arxiv_topic_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(
                arxiv_prompt_template["system"],
            ),
            MessagesPlaceholder(variable_name="history", n_messages=10),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=arxiv_prompt_template["human"],
                    input_variables=["message"],
                )
            ),
        ]
    )
