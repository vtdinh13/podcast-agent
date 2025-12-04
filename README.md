![python](https://img.shields.io/badge/Python-3.11.14-3776AB.svg?style=flat&logo=python)
![postgres](https://img.shields.io/badge/PostgreSQL-17.0-4169E1.svg?style=flat&logo=postgresql)
![docker](https://img.shields.io/badge/Docker-28.5.1-2496ED.svg?style=flat&logo=docker)
![elasticsearch](https://img.shields.io/badge/Elasticsearch-9.1.1-005571.svg?style=flat&logo=elasticsearch)
![kibana](https://img.shields.io/badge/Kibana-9.1.1-005571.svg?style=flat&logo=elasticsearch)
![pydantic](https://img.shields.io/badge/Pydantic-2.12.2-E92063.svg?style=flat&logo=pydantic)
![brave](https://img.shields.io/badge/Brave-FB542B.svg?style=flat&logo=brave)
![qdrant](https://img.shields.io/badge/Qdrant-v1.16-DC244C.svg?style=flat&)

## Introduction
The Huberman Lab podcast consistently ranks in the top 5 across Apple and Spotify in the Health, Fitness, and Science categories, with over seven million YouTube subscribers. While unequivocally popular, the episodes are long and often difficult to digest. Each episode averages 120 minutes, and the longest episode, featuring Dr. Andy Galpin on "Optimal Protocols to Build Strength and Muscle", runs 279 minutes  -- that's more than 4.5 hours!

The podcast offers evidenced-based insights and practical tools, but that information is hidden in excessively long episodes. This project addresses this gap by building an agentic system that acts as a personalized coach, surfaces relevant knowledge from the podcast archive, searches the web for the latest research on a requested topic, and recommends actionable takeaways. By grounding responses in expert interviews and the latest research, the agent can offer guidance and recommendations that are both scientifically sound and immediately actionable.

This repository introduces the Podcast Research AI Agent. Audio files are downloaded via RSS, transcribed with Faster Whisper, chunked with a sliding window, and embedded with Hugging Face's Sentence Transformer model `all-MPNet-base-v2`. The Streamlit UI utilizes the Qdrant cloud, but Elasticsearch was explored and code to use the local database remains available.

The agent is implemented with the Pydantic's BaseModel for strict Python data validation and PydanticAI's Agent class for structured output and agent tooling. OpenAI's `gpt-4o-mini` powers the reasoning and the tools given to the agent include: searching the knowledge base, retrieving recent research articles, and summarizing the current state of the research for a requested topic.

Testing and evaluation combine vibe checks, unit tests (pytest), and an LLM judge.

The diagram below outlines the development flow and supporting services.

<img src='diagrams/podcast-research-agent-system.png'>


## Setup
1. `uv` manages Python packages and the virtual environment. To replicate the project you can use either `uv` or `pip`, but using `uv` will match this repository’s workflow most closely.

    *Option 1*: Manage with uv  
      - Install `uv` if it is not already on your system. See the [Astral documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation steps.  
      - Run `uv sync` to install all required packages.
      
    *Option 2*: Manage with pip  
      - Run `pip install -r requirements.txt`.
2. Docker Desktop runs the Docker Engine daemon.
    - Download Docker Desktop if needed (refer to the [Docker Desktop docs](https://docs.docker.com/desktop/)).
    - Start Docker Desktop so the Docker Engine daemon is available.
    - Run `docker-compose up` to start every service, or `docker-compose up -d` to run them in detached mode so the containers stay in the background while you continue working in the terminal.
3. API keys are required.
    
    *Required keys*

    - The agent uses OpenAI models. Sign up for an [OpenAI API key](https://auth.openai.com/create-account) if you don't already have one. 
    - The Brave API powers the web search tool. Register for a [Brave API key](https://api-dashboard.search.brave.com/register).

    *Optional keys*
      
      - Local vector databases are sufficient, but if you want to upload embeddings to the cloud, generate keys from [Elasticsearch](https://cloud.elastic.co/registration?pg=global&plcmt=nav&cta=205352-primary) or [Qdrant](https://login.cloud.qdrant.io/u/signup/identifier?state=hKFo2SBfRTd1VlpiZHlTRFJ5a1NoUGp4T20yenJDSzhsUHI4baFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIGsxZ1RDOUc0U2UxMlNjNkdWbktLcXBneEM0em9WMlNJo2NpZNkgckkxd2NPUEhPTWRlSHVUeDR4MWtGMEtGZFE3d25lemc). 


4. API keys are managed via `direnv`. Keys live in a `.env` file, and `.envrc` contains `dotenv` so the values load automatically. Example:

    ```.env 
    OPENAI_API_KEY=openai_api_key_value
    BRAVE_API_KEY=brave_api_key_value
    ```


## Ingestion

0. Downloading and transcribing transcripts is a project on its own. This repository ships with a Parquet file of transcripts. See [Ingestion](ingestion/README.md) if you'd like to replicate the end-to-end download and transcription flow yourself.

1. Make sure Docker Desktop is running.

2. Two vector database options are available: Elasticsearch and Qdrant. The `docker-compose` file supports both. You can run both simultaneously, but picking the database that best fits your workflow and removing the other keeps resource usage light. Start every service in detached mode with `docker-compose up -d` or start just the service you selected.

3. If you choose Elasticsearch, start both the Elasticsearch and Kibana services.
    ```
    docker-compose up elasticsearch kibana -d
    ```
  
    *Optional*: Access the Kibana dashboard at http://localhost:5601. It is a helpful UI for inspecting embeddings, but the agent can function without it.

4. Ingestion with Elasticsearch

    *Skip steps 7-8 if you stay on Elasticsearch.*


  
7. If you choose Qdrant, start the service:

    ```
    docker-compose up qdrant -d
    ```
  
   *Optional*: Access the Qdrant dashboard at http://localhost:6333/dashboard.

8. Chunking and uploading embeddings to the local Qdrant vector database takes ~2 hours. Run in the CLI:

    ```
    python ingestion/qdrant.py \
      --parquet-path transcripts/transcripts.parquet \
      --collection-name transcripts \
      --distance cosine \
      --target local 
    ```

    <u>*Recommendation*</u>: Add the `--limit` argument to process only a sample of rows (each row corresponds to one episode/transcript). For example, `--limit 100` chunks and uploads the first 100 transcripts and finishes in about 40 minutes. 

      ```
      python ingestion/qdrant.py \
        --parquet-path transcripts/transcripts.parquet \
        --collection-name transcripts \
        --distance cosine \
        --target local \
        --limit 100
      ```

    *Optional*: You will be able to see your data on the Qdrant dashboard by pasting in your browser: http://localhost:6333/dashboard. 

    <img src=diagrams/qdrant_dashboard.png>
## Agent


## Test and Evaluation

## Logging and Monitoring
