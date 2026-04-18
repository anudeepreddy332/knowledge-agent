"""
Knowledge Agent Loop: (aka ReAct loop / Reason + Act loop)
    1. Takes user input
    2. Sends it to Deepseek with the tool schemas
    3. If Deepseek wants to call a tool -> execute tool schemas and feed result back
    4. Repeat until Deepseek emits a final text response (no tool call)
    5. Print and loop back for the next user turn
"""
import os
import json
from src.knowledge_agent.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.knowledge_agent.tools import TOOL_SCHEMAS, TOOL_EXECUTORS
from openai import OpenAI

# System prompt
# This is the single most important tuning knob for agent behaviour.
# It tells the model:
#   - What it is
#   - What tools it has and when to use each one
#   - What to do when it doesn't know something

SYSTEM_PROMPT = """You are a knowledgeable assistant with access to two tools:

- query_knowledge_base: searches a local vector database built from personal notes and documents.
- web_search: searches the live web via Tavily.

CALL COUNT LIMITS — these govern how many times each tool may be called, not which tool to call first.
Routing order is determined by rules 1–5 below:
- query_knowledge_base: call at most ONCE per turn. If it returns results — at any distance —
  do not call it again with a rephrased or narrower query.
- verify_claim: call at most ONCE per source chunk. All claims for a chunk go in one list.
- After query_knowledge_base returns with distance < 0.65: proceed to verify_claim, then answer.
- After query_knowledge_base returns with ALL distances > 0.65: follow rule 3 — fall back to
  web_search. Do not call query_knowledge_base again.

Routing rules — follow these precisely:
1. If the question is about a concept, explanation, or topic likely covered in personal notes
   (e.g. ML concepts, RAG, embeddings, chunking, agent architecture) → call query_knowledge_base FIRST.
1b. The knowledge base contains notes on these specific topics: RAG, embeddings, ChromaDB,
    HNSW indexing, chunking strategies (paragraph, fixed-size, semantic), sentence transformers,
    all-MiniLM-L6-v2, context poisoning, hallucination prevention, and retrieval misses.
    IMPORTANT: If the question mentions any of these topics by name, you MUST call
    query_knowledge_base FIRST — regardless of how the question is phrased.
    This includes 'how do I...', 'why does...', 'is X effective for...', 'what training
    objective...', 'why does [software product] use...', and comparative/benchmarking questions
    about models or tools that are in the topic list above.
    The topic list takes priority over your judgment about whether a question sounds like
    it needs web documentation or benchmarks.
2. If query_knowledge_base returns results with relevance score > 0.0, use those
   results to answer. Cite the source filename and chunk index in your response.
   Negative relevance scores indicate the chunk is likely irrelevant — fall back
   to web_search in that case.
2b. For every factual claim from a KB result, cite the source inline:
    (Source: filename, chunk N). Do not answer from KB results without citing.
2c. After query_knowledge_base returns, call verify_claim EXACTLY ONCE per source chunk
    used in your answer. Pass ALL claims from that chunk in a single call as a list.
    Do not call verify_claim more than once per chunk. Do not retry with rephrased claims.
    If a claim comes back NOT SUPPORTED, drop it from your answer and use what remains.
    If no claims are supported, say "I found a chunk but could not verify the details."
    If verify_claim returns NOT SUPPORTED, do not include that claim in your answer.
3. If query_knowledge_base returns nothing useful (all distances > 0.65, or content is clearly off-topic)
   → fall back to web_search and note that the answer came from the web.
4. If the question is explicitly about current events, recent news, live data, prices, or anything
   time-sensitive → skip query_knowledge_base and go straight to web_search. ALSO use web_search for general factual
   or world knowledge questions (geography, capitals, history, science facts) that are
   clearly NOT about the topics in rule 1b.
5. If both tools return nothing useful → say "I don't know" clearly. Do not fabricate.
6. The HARD LIMITS above are absolute. After one query_knowledge_base call returns,
   your only valid next actions are: (a) call verify_claim on the retrieved chunks,
   or (b) answer directly. Calling query_knowledge_base a second time in the same turn
   is always wrong — even if the phrasing is different.
7. Keep answers concise and grounded in retrieved content. Do not pad.
8. After a web_search returns useful factual information about a topic you are
   likely to be asked about again, call save_to_knowledge_base to persist it.
   Do not save: news, prices, sports results, or anything time-sensitive.
   Do save: explanations, definitions, technical details, research findings.
"""

# Tool dispatcher
# When the LLM emits a tool_use block, it gives us:
#   - tool_name: "query_knowledge_base"
#   - tool_input: {"query": "...", "n_results": 3}
#
# We look up the executor by name and call it with **tool_input (kwargs unpacking).
# The result is a plain string that goes back to the model as a tool_result message.



def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return  f"[error] Unknown tool: '{tool_name}'. Available tools: {list(TOOL_EXECUTORS.keys())}"
    try:
        return executor(**tool_input)
    except TypeError as e:
        return f"[error] Tool '{tool_name}' called with bad arguments: {e}"


# Agent loop (single turn)
# Handles one full reasoning chain (multiple tool calls before a final answer) for a user message.
# Message thread structure
# [user msg] -> [assistant: tool_use] -> [user: tool_result] -> [assistant: text]

# We keep appending to `messages` so the model has the full context of what
# it retrieved and what it reasoned about.

# The loop exits when stop_reason == "end_turn" — meaning the model has decided
# it has enough information and is giving a final answer, not requesting another tool.

# Vulnerability: infinite loop if the model keeps calling tools and never emits
# end_turn. Guard: MAX_TOOL_ROUNDS.

MAX_TOOL_ROUNDS = 8

def run_agent_turn(client: OpenAI, messages: list[dict]) -> str:
    """
    DeepSeek: uses chat.completions.create with tools.
    Mutates 'messages' in place (appends assistant and tool_result entries).
    Returns the final assistant text response.
    """

    kb_called = False

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=4000,
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message

        messages.append(msg.model_dump(exclude_none=True)) #Keep full context

        # If no tool calls, we're done
        if not msg.tool_calls:
            return msg.content or ""

        # Process tool calls
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

            # GATE: intercept redundant KB calls before they consume a round
            if tool_name == "query_knowledge_base" and kb_called:
                result_text = (
                    "[SYSTEM GATE] query_knowledge_base already called this turn. "
                    "You have retrieved chunks above — use them. "
                    "Your next step must be verify_claim on those chunks, then answer."
                )
                print(f" → BLOCKED redundant {tool_name}({tool_args})")

            else:
                if tool_name == "query_knowledge_base":
                    kb_called = True # mark first kb call
                print(f"  → calling {tool_name}({tool_args})")
                result_text = dispatch_tool(tool_name, tool_args)

            # Append tool result message with matching tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    return "[max tool rounds reached — agent did not produce a final answer]"



def main():
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL
    )
    print("Knowledge Agent ready. Type 'quit' to exit.\n")

    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        answer = run_agent_turn(client, messages)
        print(f"\nAgent: {answer}\n")

if __name__ == "__main__":
    main()

