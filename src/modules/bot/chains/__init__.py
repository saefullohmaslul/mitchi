from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_groq import ChatGroq

from src.modules.bot.parsers.chatbot_parser import string_parser
from src.modules.bot.prompts.arxiv_topic_prompt import arxiv_topic_prompt
from src.modules.bot.prompts.background_information_prompt import (
    background_information_prompt,
)
from src.modules.bot.prompts.citation_and_referencing_prompt import (
    citation_and_referencing_prompt,
)
from src.modules.bot.prompts.data_analysis_advice_prompt import (
    data_analysis_advice_prompt,
)
from src.modules.bot.prompts.ethical_considerations_prompt import (
    ethical_considerations_prompt,
)
from src.modules.bot.prompts.general_academic_inquiry_prompt import (
    general_academic_inquiry_prompt,
)
from src.modules.bot.prompts.goodbye_prompt import goodbye_prompt
from src.modules.bot.prompts.greeting_prompt import greeting_prompt
from src.modules.bot.prompts.intent_classifier_prompt import intent_classifier_prompt
from src.modules.bot.prompts.methodology_guidance_prompt import (
    methodology_guidance_prompt,
)
from src.modules.bot.prompts.other_prompt import other_prompt
from src.modules.bot.prompts.research_problem_clarification_prompt import (
    research_problem_clarification_prompt,
)
from src.modules.bot.prompts.research_prompt import research_prompt
from src.modules.bot.prompts.research_proposal_assistance_prompt import (
    research_proposal_assistance_prompt,
)
from src.modules.bot.prompts.timeline_planning_prompt import timeline_planning_prompt
from src.modules.bot.tools.arxiv_tool import get_research_paper


def intent_classifier() -> RunnableSerializable:
    """Create a chain for the intent classifier.

    Returns:
        RunnableSerializable: Chain for the intent classifier.
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, stop_sequences=["\n"])

    return (
        {
            "message": lambda x: x["message"],
        }
        | intent_classifier_prompt()
        | llm
        | string_parser
    )


def other_chain() -> RunnableSerializable:
    """Create a chain for the other intent.

    Returns:
        RunnableSerializable: Chain for the other intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.8, stop_sequences=["\n"])

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
        }
        | other_prompt()
        | llm
        | StrOutputParser()
    )


def arxiv_topic_chain() -> RunnableSerializable:
    """Create a chain for the research topic intent.

    Returns:
        RunnableSerializable: Chain for the research topic intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=["\n"])

    return (
        {
            "message": lambda x: x["message"],
        }
        | arxiv_topic_prompt()
        | llm
        | string_parser
    )


def research_topic_chain() -> RunnableSerializable:
    """Create a chain for the research intent.

    Returns:
        RunnableSerializable: Chain for the research intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": arxiv_topic_chain() | get_research_paper,
        }
        | research_prompt()
        | llm
        | StrOutputParser()
    )


def methodology_guidance_chain() -> RunnableSerializable:
    """Create a chain for the methodology guidance intent.

    Returns:
        RunnableSerializable: Chain for the methodology guidance intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | methodology_guidance_prompt()
        | llm
        | StrOutputParser()
    )


def research_problem_clarification_chain() -> RunnableSerializable:
    """Create a chain for the research problem clarification intent.

    Returns:
        RunnableSerializable: Chain for the research problem clarification intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | research_problem_clarification_prompt()
        | llm
        | StrOutputParser()
    )


def background_information_chain() -> RunnableSerializable:
    """Create a chain for the background information intent.

    Returns:
        RunnableSerializable: Chain for the background information intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | background_information_prompt()
        | llm
        | StrOutputParser()
    )


def research_proposal_assistance_chain() -> RunnableSerializable:
    """Create a chain for the research proposal assistance intent.

    Returns:
        RunnableSerializable: Chain for the research proposal assistance intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | research_proposal_assistance_prompt()
        | llm
        | StrOutputParser()
    )


def data_analysis_advice_chain() -> RunnableSerializable:
    """Create a chain for the data analysis advice intent.

    Returns:
        RunnableSerializable: Chain for the data analysis advice intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | data_analysis_advice_prompt()
        | llm
        | StrOutputParser()
    )


def citation_and_referencing_chain() -> RunnableSerializable:
    """Create a chain for the citation and referencing intent.

    Returns:
        RunnableSerializable: Chain for the citation and referencing intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | citation_and_referencing_prompt()
        | llm
        | StrOutputParser()
    )


def timeline_planning_chain() -> RunnableSerializable:
    """Create a chain for the timeline planning intent.

    Returns:
        RunnableSerializable: Chain for the timeline planning intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
        }
        | timeline_planning_prompt()
        | llm
        | StrOutputParser()
    )


def ethical_considerations_chain() -> RunnableSerializable:
    """Create a chain for the ethical considerations intent.

    Returns:
        RunnableSerializable: Chain for the ethical considerations intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | ethical_considerations_prompt()
        | llm
        | StrOutputParser()
    )


def general_academic_inquiry_chain() -> RunnableSerializable:
    """Create a chain for the general academic inquiry intent.

    Returns:
        RunnableSerializable: Chain for the general academic inquiry intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
        }
        | general_academic_inquiry_prompt()
        | llm
        | StrOutputParser()
    )


def greeting_chain() -> RunnableSerializable:
    """Create a chain for the greeting intent.

    Returns:
        RunnableSerializable: Chain for the greeting intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
        }
        | greeting_prompt()
        | llm
        | StrOutputParser()
    )


def goodbye_chain() -> RunnableSerializable:
    """Create a chain for the goodbye intent.

    Returns:
        RunnableSerializable: Chain for the goodbye intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
        }
        | goodbye_prompt()
        | llm
        | StrOutputParser()
    )


def intent_routing(info: dict) -> RunnableSerializable:
    """Route the intent to the appropriate chain.

    Args:
        info (dict): Chatbot input information.

    Returns:
        RunnableSerializable: Runnable chain based on the intent.
    """
    intent = info.get("intent", "").lower()

    intent_map = {
        "other": other_chain,
        "research_topic": research_topic_chain,
        "methodology_guidance": methodology_guidance_chain,
        "research_problem_clarification": research_problem_clarification_chain,
        "background_information": background_information_chain,
        "research_proposal_assistance": research_proposal_assistance_chain,
        "data_analysis_advice": data_analysis_advice_chain,
        "citation_and_referencing": citation_and_referencing_chain,
        "timeline_planning": timeline_planning_chain,
        "ethical_considerations": ethical_considerations_chain,
        "general_academic_inquiry": general_academic_inquiry_chain,
        "greeting": greeting_chain,
        "goodbye": goodbye_chain,
    }

    return intent_map.get(intent, other_chain)()


def full_chain() -> RunnableSerializable:
    """Create a full chain for the chatbot.

    Returns:
        RunnableSerializable: Full chain for the chatbot.
    """
    return {
        "message": lambda x: x["message"],
        "history": lambda x: x["history"],
        "intent": intent_classifier(),
    } | RunnableLambda(intent_routing)
