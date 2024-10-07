from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_groq import ChatGroq

from src.modules.bot.parsers.chatbot_parser import string_parser
from src.modules.bot.prompts.intent_classifier_prompt import intent_classifier_prompt
from src.modules.bot.prompts.other_prompt import other_prompt


def intent_classifier() -> RunnableSerializable:
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


def intent_routing(info: dict) -> RunnableSerializable:
    intent = info.get("intent", "").lower()

    if intent == "other":
        return other_chain()
    else:
        return other_chain()


def full_chain() -> RunnableSerializable:
    return {
        "message": lambda x: x["message"],
        "history": lambda x: x["history"],
        "intent": intent_classifier(),
    } | RunnableLambda(intent_routing)
