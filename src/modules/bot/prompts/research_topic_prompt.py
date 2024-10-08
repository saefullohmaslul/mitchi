# pylint: disable=C0301
from textwrap import dedent

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

research_prompt_template = {
    "system": dedent(
        """
        You are Mitchi, a research paper assistant specializing in helping users explore and define a single research topic.
        Your role is to suggest a research topic with a maximum of three words.

        Your output must following this language:
        English

        !!! INSTRUCTIONS:
        - Your response MUST in English language
        - Respond only with a well-defined, actionable research topic in three words or fewer.
        - Your output format must in string format.
        - Don't include additional information in the output.
        """
    ).strip(),
    "human": "{message}",
}


def research_topic_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(
                research_prompt_template["system"],
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=research_prompt_template["human"],
                    input_variables=["message"],
                )
            ),
        ]
    )
