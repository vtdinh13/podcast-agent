![python](https://img.shields.io/badge/Python-3.11.14-3776AB.svg?style=flat&logo=python)
![postgres](https://img.shields.io/badge/PostgreSQL-17.0-4169E1.svg?style=flat&logo=postgresql)
![docker](https://img.shields.io/badge/Docker-28.5.1-2496ED.svg?style=flat&logo=docker)
![pydantic](https://img.shields.io/badge/Pydantic-2.12.2-E92063.svg?style=flat&logo=pydantic)
![brave](https://img.shields.io/badge/Brave-FB542B.svg?style=flat&logo=brave)
![qdrant](https://img.shields.io/badge/Qdrant-v1.16-DC244C.svg?style=flat&)

## Introduction
Building lasting habits is challenging without understanding the why behind them. The Habit Builder AI Agent bridges this gap by drawing on the Huberman Lab podcast archive—a top 5 podcast on Apple and Spotify with over 7 million YouTube subscribers—to transform complex scientific insights into actionable habits supported by the latest research.

This AI agent acts as a personalized coach that delivers relevant knowledge from the podcast's extensive archive, searches the web for current research, and with access to all of this knowledge, recommends actionable takeaways grounded in expert interviews and scientific evidence.

Audio files are downloaded via RSS, transcribed with Faster Whisper, chunked with a sliding window, and embedded with Hugging Face's Sentence Transformer model `all-MPNet-base-v2`. Qdrant stores embeddings and Streamlit offers a nice interface to interact with the agent. Code allows you to create a local version. For the full UI experience, go here. 

The agent is implemented with the Pydantic's BaseModel for strict Python data validation and PydanticAI's Agent class for structured output and agent tooling. OpenAI's `gpt-4o-mini` powers the reasoning and the tools given to the agent include: searching the knowledge base, retrieving recent research articles, and summarizing the current state of the research for a requested topic.

Testing and evaluation combine vibe checks, unit tests (pytest), and an LLM judge.

The diagram below outlines the development flow and supporting services.

<img src='diagrams/podcast-research-agent-system.png'>


## Setup
1. `uv` manages Python packages and the virtual environment. To replicate the project you can use either `uv` or `pip`, but using `uv` will match this repository’s workflow most closely.

    *Option 1*: Manage with uv  
      - Install `uv` if it is not already on your system. See [Astral documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation steps.  
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
      
      - The local vector databases is sufficient, but if you want to upload embeddings to Qdrant cloud, generate an API key from [Qdrant](https://login.cloud.qdrant.io/u/signup/identifier?state=hKFo2SBfRTd1VlpiZHlTRFJ5a1NoUGp4T20yenJDSzhsUHI4baFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIGsxZ1RDOUc0U2UxMlNjNkdWbktLcXBneEM0em9WMlNJo2NpZNkgckkxd2NPUEhPTWRlSHVUeDR4MWtGMEtGZFE3d25lemc). 


4. API keys are managed via `direnv`. Keys live in a `.env` file, and `.envrc` contains `dotenv` so the values load automatically. Example:

    ```.env 
    OPENAI_API_KEY=openai_api_key_value
    BRAVE_API_KEY=brave_api_key_value
    ```


## Ingestion

0. Downloading and transcribing transcripts is a project on its own. A Parquet file of transcripts is provided to avoid this step. See [Ingestion](ingestion/README.md) if you'd like to replicate the end-to-end download and transcription flow yourself.

1. Make sure Docker Desktop is running.
  
2. Start the qdrant service:

    ```
    docker-compose up qdrant -d
    ```
  
   *Optional*: Access the Qdrant dashboard at http://localhost:6333/dashboard.

3. Chunking and uploading embeddings to the local Qdrant vector database takes ~2 hours. Run in the CLI:

    ```
    python ingestion/qdrant.py \
      --parquet-path transcripts/transcripts.parquet \
      --collection-name transcripts \
      --distance cosine \
      --target local 
    ```

    <u>*Recommendation*</u>: Add the `--limit` argument to process only a sample of transcripts (each row corresponds to one episode/transcript). For example, `--limit 100` chunks and uploads the first 100 transcripts and finishes in about 40 minutes. 

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
1. Test if the vector database and agent works. Note that this version is limited.  You can only ask one question at a time. Run the following command on CLI: 
    - with uv: `uv run researchv1.py`
    - with pip: `python researchv1.py`
2. You can also run the agent locally on Streamlit. This option includes streaming parsing and continuing conversation. Run the following command on CLI:
    - with uv: `uv run streamlit run research_app.py` 
    - with pip: `python streamlit run research_app.py`

## Test and Evaluation

## Logging and Monitoring
