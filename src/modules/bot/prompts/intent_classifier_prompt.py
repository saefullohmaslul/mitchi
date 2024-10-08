from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

intent_classifier_prompt_template = {
    "system": """
        You are Mitchi, a research paper assistant. Your task is to classify the user's intent based on the following input. Choose one of the intents:
    """,
    "human": """
        Classify the user's intent based on the following input. Choose one of the intents:
        - research_topic: User is looking for assistance in defining a research topic, including suggestions for the title or research problem.
        - methodology_guidance: User needs advice on appropriate research methods or designs for their topic.
        - research_problem_clarification: User wants to clarify or formulate their research questions or hypotheses.
        - background_information: User is seeking background information or theoretical frameworks related to their topic.
        - research_proposal_assistance: User requires help in structuring or writing their research proposal.
        - data_analysis_advice: User seeks advice on data analysis techniques suitable for their research.
        - citation_and_referencing: User asks for guidance on proper citation styles or reference management.
        - timeline_planning: User wants help planning their research timeline or schedule.
        - ethical_considerations: User needs information on ethical aspects of their research.
        - general_academic_inquiry: User asks general questions about academic life or study tips.
        - greeting: User is greeting or engaging in small talk.
        - goodbye: User is ending the conversation or expressing gratitude.
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
            MessagesPlaceholder(variable_name="history", n_messages=10),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=intent_classifier_prompt_template["human"],
                    input_variables=["message"],
                )
            ),
        ]
    )
