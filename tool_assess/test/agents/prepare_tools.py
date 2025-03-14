# from langchain_community.tools import DuckDuckGoSearchResults
#
# duck_duck_go_search = DuckDuckGoSearchResults(num_results=2)
#
# available_tools = [duck_duck_go_search]
import json

from config import settings

with open(settings.SUMMARIZED_TOOLS_PATH, 'r', encoding='utf-8') as jf:
    available_tools = json.load(jf)

