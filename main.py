import os
import jwt
from jwt import PyJWKClient
from typing import TypedDict, Literal, Optional
import psycopg2
from psycopg2 import pool
from pymilvus import connections, Collection
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# 1. Expanded State Definition
class AgentState(TypedDict):
    query: str
    token: str  # The raw OAuth2 Bearer token passed in at invocation
    user_email: Optional[str]
    route: Literal["sql_agent", "rag_agent", "unknown", "unauthorized"]
    agent_raw_data: str
    final_response: str

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize thread-safe connection pool globally for the application lifecycle
try:
    db_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=20,
        dbname=os.getenv("POSTGRES_DB", "enterprise_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
except Exception as e:
    print(f"CRITICAL: Failed to initialize database connection pool: {e}")
    db_pool = None

# 2. Authentication Node
def auth_node(state: AgentState) -> dict:
    """Verifies the OAuth2 Bearer token using JWKS and extracts the user's email."""
    token = state.get("token", "")
    try:
        jwks_url = os.environ.get("OAUTH2_JWKS_URL", "https://your-idp-domain.com/.well-known/jwks.json")
        audience = os.environ.get("OAUTH2_AUDIENCE", "your-api-audience")
        
        jwks_client = PyJWKClient(jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            options={"verify_exp": True}
        )
        
        email = payload.get("email")
        if not email:
            raise ValueError("Token does not contain an email claim.")
            
        return {"user_email": email}
    except Exception as e:
        print(f"Authentication failed: {e}")
        return {"user_email": None, "route": "unauthorized", "agent_raw_data": "Invalid or expired token."}

# 3. Intent Router Node (Modified to respect prior auth failures)
def router_node(state: AgentState) -> dict:
    if state.get("route") == "unauthorized":
        return {} # Pass-through if already failed
        
    prompt = f"""You are an elite intent router. Categorize this user request into one of two paths:
    - 'sql_agent': If the user is asking for specific counts, metrics, raw inventories, or exact lists from relational data fields like products, stock quantities, prices, or categories.
    - 'rag_agent': If the user is asking about semantic rules, abstract concepts, timelines, support clauses, policies, guidelines, or non-tabular instructions.
    
    Request: {state['query']}
    Return exactly one string option: 'sql_agent' or 'rag_agent'."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    ).choices[0].message.content.strip()
    
    return {"route": response if response in ["sql_agent", "rag_agent"] else "unknown"}

# 4. RBAC Authorization Node
def rbac_node(state: AgentState) -> dict:
    if state.get("route") == "unauthorized":
        return {}
        
    email = state.get("user_email")
    route = state.get("route")
    
    # Define restricted access lists
    no_sql_users = {"tom@ex.com", "dick@ex.com", "harry@ex.com"}
    no_rag_users = {"paul@ex.com", "ben@ex.com"}
    
    # Enforce database isolation
    if route == "sql_agent" and email in no_sql_users:
        return {
            "route": "unauthorized", 
            "agent_raw_data": f"RBAC Denied: User {email} lacks PostgreSQL read privileges."
        }
        
    # Enforce vector DB isolation
    if route == "rag_agent" and email in no_rag_users:
        return {
            "route": "unauthorized", 
            "agent_raw_data": f"RBAC Denied: User {email} lacks Milvus RAG privileges."
        }
        
    return {} # Authorized, proceed normally

# 5. Route Execution Bridge
def route_decision(state: AgentState) -> str:
    return state.get("route", "unknown")

# 6. Terminal Unauthorized Node
def unauthorized_node(state: AgentState) -> dict:
    reason = state.get("agent_raw_data", "Authentication failed.")
    return {"final_response": f"Access Denied. {reason}"}

# Agent 1: Text-to-SQL Node
def sql_agent_node(state: AgentState) -> dict:
    db_schema = "Table: products\nColumns: product_id (INT), name (VARCHAR), category (VARCHAR), price (NUMERIC), stock_quantity (INT)"
    
    prompt = f"""Given this PostgreSQL schema:\n{db_schema}\n\nGenerate an executable, valid raw SQL query to resolve this request: '{state['query']}'. 
    Provide ONLY the raw SQL string block. Do not wrap it in markdown formats."""
    
    sql_query = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    ).choices[0].message.content.strip().replace("```sql", "").replace("```", "")
    
    if not db_pool:
        return {"agent_raw_data": "System Error: Database connection pool is unavailable."}
        
    conn = None
    try:
        # Check out an active connection from the thread-safe pool
        conn = db_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            records = cursor.fetchall()
            
        # Optional: ensure clean transaction boundary
        conn.commit()
        data_str = f"SQL Executed: {sql_query}\nFetched Rows: {str(records)}"
    except Exception as e:
        if conn:
            conn.rollback()
        data_str = f"SQL execution failed: {str(e)}"
    finally:
        if conn:
            # CRITICAL: Always return the connection to the pool to prevent resource exhaustion
            db_pool.putconn(conn)
            
    return {"agent_raw_data": data_str}

# Agent 2: Milvus RAG Vector Node
def rag_agent_node(state: AgentState) -> dict:
    try:
        connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
        collection = Collection("knowledge_base")
        collection.load()
        
        emb_res = client.embeddings.create(model="text-embedding-3-small", input=[state['query']])
        vector = emb_res.data[0].embedding
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[vector], 
            anns_field="embedding", 
            param=search_params, 
            limit=2, 
            output_fields=["text"]
        )
        
        retrieved_texts = [hit.entity.get('text') for hit in results[0]]
        context_str = "\n".join(retrieved_texts)
    except Exception as e:
        context_str = f"Vector search infrastructure failure: {str(e)}"
        
    return {"agent_raw_data": context_str}

# Synthesizer Response Node
def synthesis_node(state: AgentState) -> dict:
    prompt = f"""You are a master synthesis assistant. Combine the original query with the exact database structural information retrieved below to answer the user comprehensively.
    
    User Request: {state['query']}
    Retrieved Context Block: {state['agent_raw_data']}
    
    Answer clearly and elegantly."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
    
    return {"final_response": response}

# 7. Compile the Auth-Secured Graph
def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("auth", auth_node)
    workflow.add_node("router", router_node)
    workflow.add_node("rbac", rbac_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("unauthorized_agent", unauthorized_node)
    workflow.add_node("synthesizer", synthesis_node)
    
    # Define Linear Prep Pipeline
    workflow.set_entry_point("auth")
    workflow.add_edge("auth", "router")
    workflow.add_edge("router", "rbac")
    
    # Define Conditional Routing
    workflow.add_conditional_edges(
        "rbac",
        route_decision,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "unauthorized": "unauthorized_agent",
            "unknown": "unauthorized_agent"
        }
    )
    
    # Terminal Connections
    workflow.add_edge("sql_agent", "synthesizer")
    workflow.add_edge("rag_agent", "synthesizer")
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("unauthorized_agent", END)
    
    return workflow.compile()

app = build_workflow()