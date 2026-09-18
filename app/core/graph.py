"""
DEPRECATED (v1 architecture) — superseded by app/core/graph_v2.py.

The SCAI QC report showed this free-form Text-to-SQL + LLM-does-arithmetic
pattern produces wrong numbers (see README "QC findings -> fix" table).
Kept here for reference only. The only place a v1-style free-form LLM SQL
call might still be reasonable is pure qualitative text lookups (sectors/
entities/articles) where no arithmetic or invented-number risk exists —
even there, prefer extending app/db/retriever.py with a fixed parameterized
query instead if you can.
"""

"""
Orchestration graph. Uses LangGraph as a deterministic state machine rather
than letting agents call each other freely — for economic data you want a
predictable, auditable path from question -> SQL -> data -> verified answer,
not an open-ended agent loop that might take unpredictable detours.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.agents import router_agent, sql_agent, analyst_agent, chart_agent, verifier_agent


class GraphState(TypedDict):
    user_message: str
    conversation_context: str
    intent: str
    wants_chart: bool
    sql: Optional[str]
    rows: list
    draft_answer: Optional[str]
    final_answer: Optional[str]
    chart_spec: Optional[dict]
    verified: bool
    issue: Optional[str]


def route_node(state: GraphState) -> GraphState:
    result = router_agent.classify(state["user_message"])
    state["intent"] = result["intent"]
    state["wants_chart"] = result.get("wants_chart", False)
    return state


def sql_node(state: GraphState) -> GraphState:
    result = sql_agent.generate_and_run(state["user_message"], state["conversation_context"])
    state["sql"] = result["sql"]
    state["rows"] = result["rows"]
    return state


def analyst_node(state: GraphState) -> GraphState:
    state["draft_answer"] = analyst_agent.analyze(state["user_message"], state["rows"])
    return state


def chart_node(state: GraphState) -> GraphState:
    state["chart_spec"] = chart_agent.build_chart_spec(state["user_message"], state["rows"])
    return state


def verify_node(state: GraphState) -> GraphState:
    result = verifier_agent.verify(state["rows"], state["draft_answer"])
    state["verified"] = result["verified"]
    state["issue"] = result.get("issue")

    if result["verified"]:
        state["final_answer"] = state["draft_answer"]
    else:
        # Don't silently ship an unverified number — degrade gracefully
        state["final_answer"] = (
            "I found some data but couldn't fully verify the figures in my draft answer "
            f"({result.get('issue', 'unspecified issue')}). Here is what the data shows directly:\n\n"
            + "\n".join(str(r) for r in state["rows"][:20])
        )
    return state


def general_chat_node(state: GraphState) -> GraphState:
    from app.core.llm_client import chat, llm_client
    from app.core.config import settings
    state["final_answer"] = chat(
        client=llm_client,
        model=settings.LLM_MODEL_NAME,
        system="You are SCAI's economic data assistant for Qatar. Be brief and helpful. "
               "If asked what you can do, mention you answer questions about Qatari economic "
               "indicators (GDP, inflation, trade, labor market, sectors) using SCAI data.",
        user=state["user_message"],
    )
    return state


def out_of_scope_node(state: GraphState) -> GraphState:
    state["final_answer"] = (
        "I'm scoped to Qatari economic data from SCAI (GDP, inflation, trade, sectors, "
        "labor market, etc.). That question is outside what I can help with here."
    )
    return state


def route_decision(state: GraphState) -> str:
    if state["intent"] in ("data_lookup", "trend_analysis", "comparison", "chart_request"):
        return "sql"
    elif state["intent"] == "general_chat":
        return "general_chat"
    else:
        return "out_of_scope"


def post_sql_decision(state: GraphState) -> str:
    return "chart" if state["wants_chart"] and state["rows"] else "analyst_only"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("sql", sql_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("chart", chart_node)
    graph.add_node("verify", verify_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges("route", route_decision, {
        "sql": "sql",
        "general_chat": "general_chat",
        "out_of_scope": "out_of_scope",
    })

    # Always analyze; chart runs alongside/after when requested
    graph.add_edge("sql", "analyst")
    graph.add_conditional_edges("analyst", post_sql_decision, {
        "chart": "chart",
        "analyst_only": "verify",
    })
    graph.add_edge("chart", "verify")

    graph.add_edge("verify", END)
    graph.add_edge("general_chat", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


compiled_graph = build_graph()
