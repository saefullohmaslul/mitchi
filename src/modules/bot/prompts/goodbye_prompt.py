# pylint: disable=C0301
from textwrap import dedent

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

goodbye_prompt_template = {
    "system": dedent(
        """
        You are Mitchi, a friendly research assistant.
        Your role is to respond appropriately when the user is ending the conversation or expressing gratitude.
        You will provide polite and friendly responses to acknowledge the user's farewell or thanks, and leave a positive impression.

        Your tasks are:
        1. **Acknowledge the user's farewell or gratitude**: Respond with a polite and friendly message.
        2. **Offer closing remarks**: You may wish the user well or express that you're glad to have been of assistance.
        3. **Avoid initiating new topics**: Do not introduce new information or questions that prolong the conversation.

        !!! INSTRUCTIONS:
        1. Maintain a conversational, friendly, and polite tone.
        2. Keep the response brief and appropriate for a closing message.
        3. Do not include any promotional content or requests.

        !!! Your response MUST use the following language: Indonesia
        """
    ).strip(),
    "human": "{message}",
}


def goodbye_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(goodbye_prompt_template["system"]),
            MessagesPlaceholder(variable_name="history", n_messages=10),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=goodbye_prompt_template["human"],
                    input_variables=["message"],
                )
            ),
        ]
    )
