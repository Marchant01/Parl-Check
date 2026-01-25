import requests

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
def get_documents(**kwargs):
    params = {"utformat": "json", "person_id": kwargs.get("member_id")}

    url = base_url + endpoints["anföranden"]
    response = requests.get(url, params=params, timeout=10)
    return response.json()
