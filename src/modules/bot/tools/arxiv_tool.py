import json

from arxiv import Client, Search, SortCriterion


def get_research_paper(query: str) -> str:
    """Search for research papers based on the user's query.

    Args:
        query (str): User's query.

    Returns:
        str: Research papers in JSON string format.
    """
    client = Client(page_size=10)
    search = Search(query=query, max_results=10, sort_by=SortCriterion.SubmittedDate)

    research = []
    for r in client.results(search):
        research.append(
            {
                "title": r.title,
                "summary": r.summary,
                "author": [a.name for a in r.authors],
            }
        )

    return "\n\n".join([json.dumps(doc) for doc in research])
