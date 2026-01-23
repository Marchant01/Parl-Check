import requests

base_url = "https://data.riksdagen.se"

endpoints = {
    "debatter": "/dokumentlista/?doktyp=debatt",
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

def get_debate_text(document_id, debate_nr):
    url = base_url + endpoints["anföranden"] + document_id + "-" + debate_nr + "/json"
    
    response = requests.get(url, timeout=10)
    return response.json()

def get_documents(**kwargs):
    # for documents from a specific member you need the ID, for the
    params = {"utformat": "json", "person_id": kwargs.get("member_id")}

    url = base_url + endpoints["anföranden"]
    response = requests.get(url, params=params, timeout=10)
    return response.json()
