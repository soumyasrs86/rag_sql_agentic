import os
import psycopg2
import sqlglot
from sqlglot import exp
from typing import TypedDict, Literal
from pymilvus import connections, Collection
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# State Definition
class AgentState(TypedDict):
    query: str
    route: Literal["sql_agent", "rag_agent", "unknown"]
    agent_raw_data: str
    final_response: str

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Intent Router Node
def router_node(state: AgentState) -> dict:
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

def route_decision(state: AgentState) -> str:
    return state["route"]

# Agent 1: Secure Text-to-SQL Node
def sql_agent_node(state: AgentState) -> dict:
    db_schema = "Table: products\nColumns: product_id (INT), name (VARCHAR), category (VARCHAR), price (NUMERIC), stock_quantity (INT)"
    
    prompt = f"""Given this PostgreSQL schema:\n{db_schema}\n\nGenerate an executable, valid raw SQL query to resolve this request: '{state['query']}'. 
    Provide ONLY the raw SQL string block. Do not wrap it in markdown formats."""
    
    raw_sql = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    ).choices[0].message.content.strip().replace("```sql", "").replace("```", "")
    
    try:
        # 1. AST-based parsing and validation
        parsed_query = sqlglot.parse_one(raw_sql, read="postgres")
        
        if not isinstance(parsed_query, exp.Select):
            raise ValueError("Security Violation: Only SELECT statements are permitted.")
            
        # 2. Enforce a hard row limit via AST mutation
        safe_sql = parsed_query.limit(100).sql(dialect="postgres")
        
        # 3. Connect using a dedicated read-only role
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "enterprise_db"),
            user=os.getenv("POSTGRES_USER_READONLY", "db_readonly_user"),
            password=os.getenv("POSTGRES_PASSWORD_READONLY", "secure_password"),
            host=os.getenv("POSTGRES_HOST", "localhost")
        )
        
        # 4. Enforce read-only state at the driver level
        conn.set_session(readonly=True, autocommit=True)
        cursor = conn.cursor()
        
        # 5. Prevent DoS via query timeouts (5 seconds)
        cursor.execute("SET statement_timeout = '5000';")
        
        # 6. Row-Level Security (RLS) enforcement for multi-tenant data
        # Requires database policies to be set up on the 'products' table.
        tenant_id = os.getenv("CURRENT_TENANT_ID")
        if tenant_id:
            cursor.execute("SET LOCAL rls.tenant_id = %s;", (tenant_id,))
            
        # Execute the sanitized query
        cursor.execute(safe_sql)
        records = cursor.fetchall()
        
        cursor.close()
        conn.close()
        data_str = f"SQL Executed: {safe_sql}\nFetched Rows: {str(records)}"
        
    except Exception as e:
        data_str = f"SQL pipeline failure: {str(e)}"
        
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

# Compile Graph Workflow Blueprint
def build_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("synthesizer", synthesis_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent"
        }
    )
    
    workflow.add_edge("sql_agent", "synthesizer")
    workflow.add_edge("rag_agent", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

app = build_workflow()