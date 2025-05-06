import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()
apiKey = os.getenv("MISTRAL_API_KEY")
result_limit = 10

def find_basis_paper():
    papers = None
    while not papers:
        query = input('Find papers about what: ')
        if not query:
            continue

        rsp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                           headers={"x-api-key": apiKey},
                           params={"query": query, "limit": result_limit, "fields": "title, year, citations"})
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            print("No matches found, try another query")
            continue

        print(f'Found {total} results. Showing up to {result_limit}.')
        papers = results['data']
        print_papers(papers)

    selection = ""
    while not re.fullmatch('\\d+', selection):
        selection = input('Select a paper # to base recommendations on: ')
    return results['data'][int(selection)]

def print_papers(papers: str):
    for idx, paper in enumerate(papers):
        print(f"{idx}  {paper['title']} {paper['year']} {paper['citations']}")

def query_handling(topic: str=None, year: str=None, citationCount: int=None):
    # API endpoint
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"

    query = f"topic: {topic}, year: {year}, citations: {citationCount}"

    query_params = {
        # f"query": {query},
        f"fields": query}

    # Find a research paper on [topic] that was published [in/before/after] [year] and has [number of citations] citations.

    headers = {"x-api-key": apiKey} # may be wrong, has to grab api key

    # Request
    response = requests.get(url, params=query_params, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")