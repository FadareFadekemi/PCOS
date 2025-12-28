# PCOS
CystaCare is a production-grade Retrieval-Augmented Generation (RAG) assistant designed to provide evidence-based information on Polycystic Ovary Syndrome (PCOS). By combining a curated internal knowledge base with real-time medical web searching, CystaCare delivers medically grounded, empathetic, and context-aware responses to complex hormonal health queries.

🚀 Key Features
Smart Routing (Hybrid RAG):

-Internal Knowledge Base: Prioritizes a verified database of PCOS Q&A pairs stored in Supabase.

-Tavily Web Search: Automatically falls back to high-depth web searches for "latest" research or complex "mechanisms" not found in the local database.

-Persistent Chat Memory: Maintains context across multiple exchanges, allowing for follow-up questions and personalized support.

-Medical Grounding: Uses strict "Zero-Refusal" and "Forced-Grounding" prompt engineering to ensure the AI remains focused on medical data and avoids hallucinations.

-Automatic Ingestion: Seeds the vector database on startup from a local JSON source if data is missing.


- Component        |Technology
- Backend          | Framework	FastAPI
- LLM & Embeddings |OpenAI (GPT-4o-mini & Text-Embedding-3-Small)
- Vector Database	 | Supabase (pgvector)
- Web Retrieval	   |Tavily AI
- Orchestration	   |LangChain & Custom Async Logic
- Deployment       |Render| https://cystacare-frontend.onrender.com/ |   


🏗️ System Architecture
The system operates on a dual-pathway logic to ensure the user receives the most accurate information possible:

- User Query: The user asks a question via the FastAPI endpoint.

- Vector Search: The system converts the query into an embedding and performs a similarity search in Supabase.

Dynamic Decision:

-If a high-confidence match is found: Uses the Internal Database.

-If no match is found or "latest research" is requested: Triggers Tavily Web Search.

-Augmented Generation: The LLM synthesizes the retrieved context and chat history into a final response.



⚙️ Installation & Setup
1. Prerequisites
-Python 3.9+

-Supabase Account (with pgvector enabled)

-OpenAI API Key

-Tavily API Key

2. Environment Variables
-Create a .env file in the root directory:
-OPENAI_API_KEY=your_openai_key
-SUPABASE_URL=your_supabase_url
-SUPABASE_SERVICE_KEY=your_supabase_service_key
-TAVILY_API_KEY=your_tavily_key


3. Install Dependencies
-pip install -r requirements.txt

4. Database Setup
-Run the following SQL in your Supabase SQL Editor to enable vector search:
``` -- Enable the pgvector extension
create extension if not exists vector;

-- Create the documents table
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(1536)
);

-- Create a search function
create or replace function match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$; 

5. Start the Server:
```uvicorn app:app --reload ```

6. 📂 Project Structure
-app.py: Core application logic, RAG pipeline, and API endpoints.

-data/: Directory containing pcos_question_and_answer_pairs_data.json for initial seeding.

-.env: Configuration for API keys and database URLs.



