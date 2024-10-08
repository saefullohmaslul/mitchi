from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_groq import ChatGroq

from src.modules.bot.parsers.chatbot_parser import string_parser
from src.modules.bot.prompts.intent_classifier_prompt import intent_classifier_prompt
from src.modules.bot.prompts.other_prompt import other_prompt
from src.modules.bot.prompts.research_prompt import research_prompt
from src.modules.bot.prompts.research_topic_prompt import research_topic_prompt
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


def research_topic_chain() -> RunnableSerializable:
    """Create a chain for the research topic intent.

    Returns:
        RunnableSerializable: Chain for the research topic intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=["\n"])

    return (
        {
            "message": lambda x: x["message"],
        }
        | research_topic_prompt()
        | llm
        | string_parser
    )


def research_chain() -> RunnableSerializable:
    """Create a chain for the research intent.

    Returns:
        RunnableSerializable: Chain for the research intent.
    """
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3, stop_sequences=None)

    return (
        {
            "message": lambda x: x["message"],
            "history": lambda x: x["history"],
            "context": research_topic_chain() | get_research_paper,
        }
        | research_prompt()
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
        "research": research_chain,
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
