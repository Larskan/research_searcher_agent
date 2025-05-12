import requests
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
apiKey = os.getenv("MISTRAL_API_KEY")

def print_papers(papers: List[dict]) -> str:
    output = []
    for idx, paper in enumerate(papers):
        output.append(f"{idx+1}. {paper['title']} ({paper['year']}, {paper['citations']} citations)")
    final_output = "\n".join(output)
    print(final_output)
    return final_output

def query_handling(topic: str, year: str, citationCount: str) -> str:
    # API endpoint
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"

    query_params = {
        f"query": f"{topic} year:{year} citations:{citationCount}",
        f"fields": "title,year,citations",
        "limit": 5}

    headers = {"x-api-key": apiKey} 

    # Request
    response = requests.get(url, params=query_params, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
