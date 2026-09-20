import os
from typing import TypedDict,Literal
import psycopg2 as pg2
from pymilvus import connections,Collection
from opnai import OpenAI
from langgraph.graph import StateGraph,START,END



class AgentState(TypedDict):
    query:str
    route:Literal['sql_agent','rag_agent','unknown']
    final_response:str
    agent_raw_data:str

client=OpenAI(api_key=os.environ.get(OPENAI_API_KEY))

##Router node

def router_node(state:AgentState):
    prompt=f"""You are an elite intent router. Categorize this user request into one of two paths:
    - 'sql_agent': If the user is asking for specific counts, metrics, raw inventories, or exact lists from relational data fields like products, stock quantities, prices, or categories.
    - 'rag_agent': If the user is asking about semantic rules, abstract concepts, timelines, support clauses, policies, guidelines, or non-tabular instructions.
    Request:{state['query']}
    Return exactly one string option: 'sql_agent' or 'rag_agent'."""
    response=client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role':'user','content':prompt}]
    ).choices[0].message.content.strip()
    return {"route":response if response in ['sql_agent','rag_agent'] else 'unknown'}

####conditional routing
def route_decision(state:AgentState)->str:
    return state['route']



def text_to_SQL_agent(state:AgentState):
    db_schema="Table: products \n Columns:product_id(INT), name(VARCHAR), category(VARCHAR),price(NUMERIC),stock_quantity(INT)"
    prompt=f"""Given this postfre sql schema:\n {db_schema}\n\n generatr an executable,valid raw SQL queryto resolve this request:\n '{state['query']}'\n\n
     Provide ONLY the raw SQL string block. Do not wrap it in markdown formats. """
    sql_query=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
    ).choices[0].message.content.strip().replace("'''sql")
    ##connection
    try:
        conn=pg2.connect(dbname=os.getenv("POSTGRES_DB", "enterprise_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            host=os.getenv("POSTGRES_HOST", "localhost")

        )
        cursor=conn.cursor()
        cursor.execute(sql_query)
        records=cursor.fetchall()
        cursor.close()
        conn.close()
        data_str=f"sql executed:{sql_query}\n fetched rows {str(records)}"
    except Exception as e:
        data_str=f"SQL generation or execution failed : {str(e)}"
    return {"agent_raw_data":data_str}

    ######RAG Agent
def rag_agent_node(state:AgentState):
    try:
        connections.connect()
        collection=Collection('knowledgebase')
        collection.load()

        emb_res=client.embeddings.create(model="text-embedding-3-small",input=state['query'])
        vector=emb_res.data[0].embedding

        search_params={'metric_type':"L2","params":{"nprobe":10}}
        results=collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=2,
            output_fields=['text']
        )
        retrieved_texts=[hit.entity.get('text') for hit in results[0]]
        context_str="\n".join(retrieved_texts)
    except Exception as e:
        context_str=f"vector search infra failure:{str(e)}"
    return {'agent_raw_data':context_str}
####Synthesizer node

def synthesis_node(state:AgentState)->dict:
    prompt = f"""You are a master synthesis assistant. Combine the original query with the exact database structural information retrieved below to answer the user comprehensively.
    User Request: {state['query']}
    Retrieved Context Block: {state['agent_raw_data']}
    
    Answer clearly and elegantly."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
    
    return {"final_response": response}

###Graph
def build_graph():
    graph=StateGraph(AgentState)

    graph.add_node("router",router_node)
    graph.add_node("sql_agent",text_to_SQL_agent)
    graph.add_node("rag_agent",rag_agent_node)
    graph.add_node("synthesizer",synthesis_node)
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"sql_agent":'sql_agent',
         "rag_agent":"rag_agent"})
    graph.add_edge(START,"router")
    graph.add_edge("sql_agent","synthesizer")
    graph.add_edge("rag_agent","synthesizer")
    graph.add_edge('synthesizer',END)
    return graph.compile()

app=build_graph()



    
     