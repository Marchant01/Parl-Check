import requests

from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document

base_url = "https://data.riksdagen.se"

endpoints = {
    "dokument": "/dokument/",
    "voteringar": "/votering/",
    "ledamöter": "/personlista/",
    "anföranden": "/anforande/",
}


def get_members(party: str):
    parties = {
        "Socialdemokraterna": "S",
        "Moderata samlingspartiet": "M",
        "Sverigedemokraterna": "SD",
        "Miljöpartiet": "MP",
        "Centerpartiet": "C",
        "Vänsterpartiet": "V",
        "Kristendemokraterna": "KD",
        "Liberalerna": "L",
        "Ny demokrati": "nd",
    }

    url = base_url + endpoints["ledamöter"]

    params = {"utformat": "json", "parti": parties[party]}

    response = requests.get(url, params=params, timeout=10)
    print(response.json())


def get_all_members():
    # Get all members and sort them by party
    url = base_url + endpoints["ledamöter"]

    params = {"utformat": "json", "sort": "parti"}

    response = requests.get(url, params=params, timeout=10)
    return response.json()

def get_document(dok_id):
    url = base_url + endpoints["dokument"] + dok_id
    response = requests.get(url, timeout=10)
    return response.text


def get_voting(votering_id):
    url = base_url + endpoints["voteringar"] + votering_id + "/json"
    response = requests.get(url, timeout=10)
    return response.json()

# for documents from a specific member you need the ID. Not needed now
# def get_documents(**kwargs):
#     params = {"utformat": "json", "person_id": kwargs.get("member_id")}

#     url = base_url + endpoints["anföranden"]
#     response = requests.get(url, params=params, timeout=10)
#     return response.json()

def fetch_document_html(dok_id: str) -> Document:
    """
    Fetches the HTML content of a document by its ID and returns it as a Document object.
    """
    
    try:
        html = get_document(dok_id)
        return content(
            page_content=html,
            metadata={"dok_id": dok_id, "source": f"https://data.riksdagen.se/dokument/{dok_id}/html"}
        )
    except Exception as e:
        print(f"Error fetching document {dok_id}: {e}")
        error = {'Error': e}
        return error

def get_documents(dok_ids: list[str]) -> list[dict[str, str]]:
    """
    Fetches documents by their IDs and returns a list of dictionaries containing the document content.
    """
    documents = []
    bs_transformer = BeautifulSoupTransformer()
    
    try:
        if not isinstance(dok_ids, list):
            print("Invalid input: dok_ids must be a list.")
            raise TypeError("dok_ids must be a list.")
        if len(dok_ids) == 0:
            print("No document IDs provided.")
            raise ValueError("No document IDs provided.")
        else:        
            for dok_id in dok_ids:
                document = {}
                fetched_doc = fetch_document_html(dok_id)
                cleaned = bs_transformer.transform_documents([fetched_doc], tags_to_extract=["p", "h1", "h2", "h3", "li"])

                document["dok_id"] = dok_id
                document["innehåll"] = cleaned[0].page_content

                documents.append(document)

            return documents
    
    except Exception as e:
        print(f"Error fetching documents: {e}")