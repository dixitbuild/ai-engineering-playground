from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="search the web for recent information. Useful for when you need to answer questions about current events or find specific information online. Input should be a search query."
)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

# wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
def safe_wiki_query(query: str) -> str:
    try:
        return wiki_api_wrapper.run(query)
    except Exception as e:
        # print(f"Error querying Wikipedia: {e}")
        # return "No information found."
        return f"Wikipedia lookup failed ({type(e).__name__}). Try another tool or rephrase."
    
wiki_tool = Tool(
    name="wikipedia_tool",
    func=safe_wiki_query,
    description="useful for when you need to answer questions about general knowledge. Input should be a search query."
)