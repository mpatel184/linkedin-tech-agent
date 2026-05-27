import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import requests
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load environment variables from the .env file in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))

# Verify the API key is present
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY not found. Please check your .env file.")

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    raw_news: Annotated[List[dict], operator.add] 
    selected_article: dict       
    relevance_score: int         
    target_audience: str         
    draft_post: str              
    feedback: str                
    is_approved: bool            

# Pydantic schema for Gemini's structured output
class CritiqueOutput(BaseModel):
    selected_index: int = Field(description="The index of the best article in the list")
    relevance_score: int = Field(description="Score from 1-10 based on LinkedIn potential")
    target_audience: str = Field(description="Primary audience, e.g., 'Software Engineers'")

# ==========================================
# 3. NODES & ROUTING LOGIC
# ==========================================
def fetch_news_node(state: AgentState) -> dict:
    print("\n📰 [Node] Fetching Hacker News Top Stories...")
    top_ids_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    top_ids = requests.get(top_ids_url).json()
    
    fetched_articles = []
    # Fetch details for the top 5 stories on the front page
    for story_id in top_ids[:5]:
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        story_data = requests.get(story_url).json()
        
        fetched_articles.append({
            "title": story_data.get("title"),
            "url": story_data.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
            "score": story_data.get("score")
        })
    return {"raw_news": fetched_articles}

def filter_and_critique_node(state: AgentState) -> dict:
    print("🔬 [Node] Analyzing trends & selecting best topic...")
    articles = state["raw_news"]
    
    articles_text = ""
    for i, art in enumerate(articles):
        articles_text += f"[{i}] Title: {art['title']} (HN Upvotes: {art['score']})\n"
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert tech content curator. Review these Hacker News articles:\n\n"
        "{articles}\n\n"
        "Select the single best article that has broad tech appeal and high engagement "
        "potential for a professional LinkedIn post."
    )
    
    # Initialize Gemini with structured output constraints
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(CritiqueOutput)
    chain = prompt | llm
    result = chain.invoke({"articles": articles_text})
    
    return {
        "selected_article": articles[result.selected_index],
        "relevance_score": result.relevance_score,
        "target_audience": result.target_audience
    }

def writer_node(state: AgentState) -> dict:
    print("✍️ [Node] Drafting LinkedIn post with Gemini...")
    article = state["selected_article"]
    audience = state["target_audience"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are a premier tech content creator on LinkedIn. Write an engaging post based on this article:\n"
        "Title: {title}\n"
        "URL: {url}\n"
        "Target Audience: {audience}\n\n"
        "Requirements:\n"
        "1. Start with a compelling hook line to maximize 'see more' clicks.\n"
        "2. Provide 2-3 high-value technical takeaways structured with clear emojis.\n"
        "3. End with an engaging question to drive comments (Call to Action).\n"
        "4. Include 3 relevant technical hashtags.\n"
        "Keep the tone professional, insightful, and clear."
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chain = prompt | llm
    result = chain.invoke({
        "title": article["title"],
        "url": article["url"],
        "audience": audience
    })
    
    return {"draft_post": result.content}

def human_gate_node(state: AgentState) -> dict:
    print("\n=== 📢 CURRENT LINKEDIN DRAFT ===")
    print(state["draft_post"])
    print("================================\n")
    
    # Pause the graph execution and wait for your manual terminal input
    user_input = input("Satisfied? Type 'yes' to approve, or type your feedback to rewrite: ")
    
    if user_input.strip().lower() == 'yes':
        return {"is_approved": True, "feedback": ""}
    else:
        return {"is_approved": False, "feedback": user_input}

def revise_node(state: AgentState) -> dict:
    print("🔄 [Node] Revising draft based on your feedback...")
    prompt = ChatPromptTemplate.from_template(
        "You are updating a LinkedIn post based on user feedback.\n\n"
        "Original Draft:\n{draft}\n\n"
        "User Feedback: {feedback}\n\n"
        "Please rewrite the post incorporating the feedback while maintaining high quality."
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chain = prompt | llm
    result = chain.invoke({"draft": state["draft_post"], "feedback": state["feedback"]})
    
    return {"draft_post": result.content}

def success_printer_node(state: AgentState) -> dict:
    print("\n🎉 SUCCESS! COPY AND PASTE THIS TO LINKEDIN:")
    print("==========================================")
    print(state["draft_post"])
    print("==========================================\n")
    return {}

def route_after_human_gate(state: AgentState) -> str:
    # Router checks your decision flag
    if state["is_approved"]:
        return "publisher_node"
    else:
        return "revise_node"

# ==========================================
# 4. GRAPH COMPILATION
# ==========================================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("fetch_news", fetch_news_node)
workflow.add_node("filter_news", filter_and_critique_node)
workflow.add_node("writer", writer_node)
workflow.add_node("human_gate", human_gate_node)
workflow.add_node("revise", revise_node)
workflow.add_node("publisher_node", success_printer_node)

# Set up flow connections
workflow.set_entry_point("fetch_news")
workflow.add_edge("fetch_news", "filter_news")
workflow.add_edge("filter_news", "writer")
workflow.add_edge("writer", "human_gate")
workflow.add_edge("revise", "human_gate")

# Add conditional routing
workflow.add_conditional_edges(
    "human_gate",
    route_after_human_gate,
    {
        "publisher_node": "publisher_node",
        "revise_node": "revise"
    }
)
workflow.add_edge("publisher_node", END)

# Compile into an executable app
app = workflow.compile()

# ==========================================
# 5. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting the LangGraph Tech Agent...")
    initial_state = {
        "raw_news": [],
        "selected_article": {},
        "relevance_score": 0,
        "target_audience": "",
        "draft_post": "",
        "feedback": "",
        "is_approved": False
    }
    final_state = app.invoke(initial_state)
    print("🏁 Graph execution finished.")