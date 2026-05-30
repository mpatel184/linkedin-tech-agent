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
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load environment variables from the .env file in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))

# Verify that at least one LLM API key is present
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("HF_TOKEN"):
    raise ValueError("❌ Neither GOOGLE_API_KEY nor HF_TOKEN found. Please check your .env file.")

WRITER_PERSONA = (
    "You are a senior software engineer who occasionally shares genuine technical "
    "observations on LinkedIn — not a content marketer, not a thought leader. "
    "You write plainly, specifically, and without hype."
)

WRITING_RULES = (
    "STRICT RULES — follow every one of these:\n"
    "1. Write in first person, conversational tone. Like you are sharing a genuine thought, not a newsletter.\n"
    "2. Open with a short 1-2 sentence hook that feels like something a real person would say — no generic 'Have you ever felt...' openers.\n"
    "3. Give 2-3 technical takeaways as short paragraphs, NOT bullet points or headers. Weave them naturally into the text.\n"
    "4. Use at most 2-3 emojis total in the entire post. Place them inline mid-sentence, not as bullet markers.\n"
    "5. NO markdown formatting whatsoever. No ---, no ####, no **, no bullet dashes. Plain text only.\n"
    "6. End with a single genuine question to invite discussion. Keep it short and specific, not broad.\n"
    "7. Add 3 relevant hashtags on the last line.\n"
    "8. Total length: 150-220 words. Tight and readable.\n"
    "9. The last line before hashtags must be exactly: 'Full article: ' followed by the article URL. "
    "The URL will be provided to you directly — use it verbatim, do not use a placeholder."
)

WRITING_RULES_SHORT = (
    "Rules: plain text only, no bullet points, no markdown. "
    "150-200 words. End with one specific question. "
    "Last line: 'Full article: {url}' then 3 hashtags."
)

def run_llm_chain(prompt_template, input_data, structured_schema=None):
    """
    Runs a LangChain chain using Gemini 2.5 Flash as the primary model.
    Falls back to Qwen2.5-7B-Instruct if the Gemini API fails or limits are hit.
    """
    # Try Gemini first
    try:
        google_key = os.getenv("GOOGLE_API_KEY")
        if not google_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment.")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        if structured_schema:
            llm = llm.with_structured_output(structured_schema)
        chain = prompt_template | llm
        return chain.invoke(input_data)
    except Exception as e:
        print(f"\n⚠️ Gemini API limit hit or failed: {e}")
        # Reload env variables from .env to capture any runtime updates
        load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            detected_keys = [k for k in ["GOOGLE_API_KEY", "HF_TOKEN"] if os.getenv(k)]
            print(f"❌ HF_TOKEN not found in environment.")
            print(f"   Detected API keys in environment: {detected_keys}")
            print("   Please ensure you have saved your `.env` file after editing.")
            raise e
        
        # Ensure the token is set in standard Hugging Face env vars
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        
        print("🔄 Falling back to Qwen2.5-7B-Instruct via Hugging Face...")
        
        # Swap full writing rules with WRITING_RULES_SHORT for Qwen fallback
        try:
            original_template = prompt_template.messages[0].prompt.template
            if WRITING_RULES in original_template:
                new_template = original_template.replace(WRITING_RULES, WRITING_RULES_SHORT)
                prompt_template = ChatPromptTemplate.from_template(new_template)
        except Exception as t_err:
            print(f"⚠️ Failed to swap rules for Qwen: {t_err}")

        endpoint_llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=hf_token,
        )
        llm = ChatHuggingFace(llm=endpoint_llm)
        
        if structured_schema:
            # We append JSON formatting instructions to the prompt to help Qwen structure its response in JSON mode
            schema_fields = ", ".join([f"'{k}' ({v.annotation.__name__ if hasattr(v.annotation, '__name__') else str(v.annotation)})" for k, v in structured_schema.model_fields.items()])
            json_instruction = (
                f"\n\nYou MUST respond with a valid JSON object matching this schema. "
                f"Include the following JSON keys: {schema_fields}. "
                f"Do not include any explanation or markdown block, only the JSON block."
            )
            prompt_template = prompt_template + json_instruction
            llm = llm.with_structured_output(structured_schema, method="json_mode")
            
        chain = prompt_template | llm
        result = chain.invoke(input_data)
        
        # Convert dictionary to Pydantic object if using json_mode and structured schema
        if structured_schema and isinstance(result, dict):
            result = structured_schema(**result)
            
        return result


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    raw_news: Annotated[List[dict], operator.add] 
    selected_article: dict       
    relevance_score: int         
    target_audience: str         
    draft_post: str              
    feedback: Annotated[List[str], operator.add]            
    is_approved: bool            

# Pydantic schema for Gemini's structured output
class CritiqueOutput(BaseModel):
    selected_index: int = Field(description="The index of the best article in the list")
    relevance_score: int = Field(description="Score from 1-10 based on LinkedIn potential")
    target_audience: str = Field(description="Primary audience, e.g., 'Software Engineers'")

# ==========================================
# 3. NODES & ROUTING LOGIC
# ==========================================
# Constants definition moved to setup section

def fetch_news_node(state: AgentState) -> dict:
    print("\n📰 [Node] Fetching Hacker News Top Stories...")
    top_ids_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    top_ids = requests.get(top_ids_url).json()
    
    fetched_articles = []
    # Fetch details for the top 5 stories on the front page
    for story_id in top_ids[:10]:
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
        "Select the single best article based on these criteria in order:\n"
        "1. Relevance to working software engineers or engineering managers\n"
        "2. Practical insight — something that changes how someone thinks or works\n"
        "3. Timeliness — recent trends over evergreen basics\n"
        "4. Avoid: funding news, acquisition announcements, or pure research papers with no practical angle"
    )
    
    # Run chain with structured output constraints and fallback
    result = run_llm_chain(prompt, {"articles": articles_text}, CritiqueOutput)
    
    return {
        "selected_article": articles[result.selected_index],
        "relevance_score": result.relevance_score,
        "target_audience": result.target_audience
    }

def writer_node(state: AgentState) -> dict:
    print("✍️ [Node] Drafting LinkedIn post with Gemini...")
    article = state["selected_article"]
    audience = state["target_audience"]
    
    print(f"DEBUG: url being passed = {article['url']}")
    
    prompt_text = (
        f"{WRITER_PERSONA}\n\n"
        "Write a LinkedIn post based on this article:\n"
        "Title: {{title}}\n"
        "URL: {{url}}\n"
        "Target Audience: {{audience}}\n\n"
        f"{WRITING_RULES}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    result = run_llm_chain(prompt, {
        "title": article["title"],
        "url": article["url"],
        "audience": audience
    })
    
    return {"draft_post": result.content}

def human_gate_node(state: AgentState) -> dict:
    print("\n" + "="*50)
    print(state["draft_post"])
    print("="*50 + "\n")
    
    # Pause the graph execution and wait for your manual terminal input
    user_input = input("Satisfied? Type 'yes' to approve, or type your feedback to rewrite: ")
    
    if user_input.strip().lower() == 'yes':
        return {"is_approved": True}
    else:
        return {"is_approved": False, "feedback": [user_input.strip()]}

def revise_node(state: AgentState) -> dict:
    print("🔄 [Node] Revising draft based on your feedback...")
    article = state["selected_article"]
    
    print(f"DEBUG: url being passed = {article['url']}")
    
    # Format the accumulated feedback history
    feedback_history = "\n".join([f"- {fb}" for fb in state["feedback"]])
    
    prompt_text = (
        f"{WRITER_PERSONA}\n\n"
        "You are updating a LinkedIn post based on user feedback.\n\n"
        "Article Details:\n"
        "Title: {{title}}\n"
        "URL: {{url}}\n"
        "Target Audience: {{audience}}\n\n"
        "Original Draft:\n{{draft}}\n\n"
        f"Feedback History:\n{feedback_history}\n\n"
        "Please rewrite the post incorporating the feedback while strictly maintaining the rules.\n\n"
        f"{WRITING_RULES}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    result = run_llm_chain(prompt, {
        "title": article["title"],
        "url": article["url"],
        "audience": state["target_audience"],
        "draft": state["draft_post"]
    })
    
    return {"draft_post": result.content}

def send_email_node(state: AgentState) -> dict:
    print("\n📧 [Node] Sending approved post to your email...")

    sender   = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")

    if not sender or not password or not receiver or "your_email" in sender or "your_app_password" in password:
        print("\n⚠️ [Node] Email sending skipped/failed: Email environment variables not configured.")
        print("Please configure these in your `.env` file:")
        print("  EMAIL_SENDER=your_email@gmail.com")
        print("  EMAIL_PASSWORD=your_app_password")
        print("  EMAIL_RECEIVER=receiver_email@gmail.com")
        print("\n(Note: For Gmail, EMAIL_PASSWORD must be a 16-character App Password, not your standard login password.)\n")
        return {}

    # Build email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"✅ LinkedIn Post Ready — {state['selected_article']['title'][:50]}"
    msg["From"]    = sender
    msg["To"]      = receiver

    # Plain text body
    body = f"""
Your LinkedIn post is approved and ready to publish!

==========================================
{state["draft_post"]}
==========================================

Article Source : {state["selected_article"]["url"]}
Relevance Score: {state["relevance_score"]}/10
Target Audience: {state["target_audience"]}

Just copy the post above and paste it on LinkedIn.
    """

    msg.attach(MIMEText(body, "plain"))

    # Send via Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email sent to {receiver}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

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
workflow.add_node("publisher_node", send_email_node)

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
        "feedback": [],
        "is_approved": False
    }
    final_state = app.invoke(initial_state)
    print("🏁 Graph execution finished.")