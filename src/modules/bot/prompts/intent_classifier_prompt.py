from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

# - search_research_paper: User asks to search for a research paper.
# - summarize_paper: User asks to summarize a research paper.
# - list_related_papers: User asks for related papers based on user's input.
# - ask_for_paper_details: User asks for details of a research paper.
# - discuss_research: User wants to discuss a research paper.
# - request_full_paper: User asks for the full paper.

intent_classifier_prompt_template = {
    "system": """
        You are Mitchi, a research paper assistant. Your task is to classify the user's intent based on the following input. Choose one of the intents:
    """,
    "human": """
        Classify the user's intent based on the following input. Choose one of the intents:
        - research: The user is looking for assistance in defining a research topic, including suggestions for the title, objectives, or research problem.
        - other: User asks for something else.

        Input: "{message}"

        Your output must be one of the intents above and the format is string.
        Don't include additional information in the output.
    """,
}


def intent_classifier_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(
                intent_classifier_prompt_template["system"],
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=intent_classifier_prompt_template["human"],
                    input_variables=["message"],
                )
            ),
        ]
    )
