# Welcome to Parl-Check!
This project aims to let users ask questions related to the Swedish parliament based on the open documents from the parliaments API, which is then stored to a local Postgres database and generated embeddings for all tables. This let's the user add their own LLM-API key to an .env file in order to utilize RAG on the database embeddings.

The project is meant to be run fully localy meaning that the embeddings are generated with a local sentence transformer huggingface model: sentence-transformers/all-mpnet-base-v2. While the LLM is set to a gemini modelm you can switch it out to a local LLM model that supports the OpenAI format.

This project will be left behind for a future hosted application!

## Setup

### Documents and .env variables:
To generate embeddings for the tables in the databse, you need the following directory structure for the documents:

```
gov_check
│   └── documents
│       ├── anforanden
│       │   ├── anforande-202223.sql
│       │   ├── anforande-202324.sql
│       │   ├── anforande-202425.sql
│       │   └── anforande-202526.sql
│       ├── personer
│       │   └── personer
│       └── voteringar
│           ├── votering-202223.sql
│           ├── votering-202324.sql
│           ├── votering-202425.sql
│           └── votering-202526.sql
└──.env
```

All of the files can be found on the official website: https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/

All of the documents except for "personer" can be installed as .sql files except for "personer" which needs to be installed as csv.

In the .env you need to the following variables:
```
LLM_API_KEY="your-api-key"
DATABASE_URL="postgresql+psycopg2://gov_check_user:gov_check_pw@localhost:5432/gov_check_db"
```

The same connection is used for creating the database. As for the case with the api key, you can use whichever LLM provider that you desire but you need to update [chatbot.py](chatbot.py#L46) to match the provider. 

### Install python modules and setup database:
Simply install with uv:
```
# Install the python modules:
uv sync

# Create the database with docker:
docker compose run db

# Then run this script to add and seed the database with tables and embeddings
uv run database.py
```

### Running the application
To start the databse, you compose it with the same docker command as previously and run the streamlit client in your browser with:
```
# Database
docker compose run db

# Client
uv run streamlit run main.py
```