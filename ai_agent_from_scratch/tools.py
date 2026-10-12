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