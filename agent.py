import operator
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Annotated, List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load environment variables from the .env file in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))

# Verify that at least one LLM API key is present
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("HF_TOKEN"):
    raise ValueError("❌ Neither GOOGLE_API_KEY nor HF_TOKEN found. Please check your .env file.")

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
HN_REQUEST_TIMEOUT = 10
HN_ARTICLE_LIMIT = 10


class AgentPipelineError(RuntimeError):
    """User-facing error raised when an external pipeline step cannot complete."""


def extract_llm_text(result) -> str:
    """Return text from either a chat message object or a plain string response."""
    return result.content if hasattr(result, "content") else str(result)


def enforce_article_url(draft: str, article_url: str) -> str:
    """Ensure the final post uses the selected article URL, even if the LLM drifts."""
    article_line = f"Full article: {article_url}"
    lines = [line.rstrip() for line in draft.strip().splitlines()]

    if not lines:
        return article_line

    for index, line in enumerate(lines):
        if line.strip().lower().startswith("full article:"):
            lines[index] = article_line
            return "\n".join(lines).strip()

    insert_at = len(lines)
    while insert_at > 0 and not lines[insert_at - 1].strip():
        insert_at -= 1

    hashtag_start = insert_at
    while hashtag_start > 0 and lines[hashtag_start - 1].lstrip().startswith("#"):
        hashtag_start -= 1

    if hashtag_start < insert_at:
        lines.insert(hashtag_start, article_line)
    else:
        lines.extend(["", article_line])

    return "\n".join(lines).strip()


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
            raise AgentPipelineError(
                "Gemini failed and HF_TOKEN is not configured. Check GOOGLE_API_KEY, quota, or add HF_TOKEN."
            ) from e
        
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
            
        try:
            chain = prompt_template | llm
            result = chain.invoke(input_data)
        except Exception as hf_error:
            raise AgentPipelineError(
                "Both Gemini and Hugging Face generation failed. Check API keys, quota, and network connection."
            ) from hf_error
        
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
    try:
        response = requests.get(HN_TOP_STORIES_URL, timeout=HN_REQUEST_TIMEOUT)
        response.raise_for_status()
        top_ids = response.json()
    except requests.RequestException as exc:
        raise AgentPipelineError(
            "Could not fetch Hacker News stories. Check your internet connection and try again."
        ) from exc
    except ValueError as exc:
        raise AgentPipelineError("Hacker News returned an unexpected response. Try again.") from exc

    if not isinstance(top_ids, list):
        raise AgentPipelineError("Hacker News returned an unexpected story list. Try again.")

    fetched_articles = []
    for story_id in top_ids[:HN_ARTICLE_LIMIT]:
        story_url = HN_ITEM_URL.format(story_id=story_id)
        try:
            story_response = requests.get(story_url, timeout=HN_REQUEST_TIMEOUT)
            story_response.raise_for_status()
            story_data = story_response.json() or {}
        except (requests.RequestException, ValueError) as exc:
            print(f"Skipping story {story_id}: {exc}")
            continue

        if not story_data.get("title"):
            continue

        fetched_articles.append({
            "title": story_data.get("title"),
            "url": story_data.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
            "score": story_data.get("score")
        })
    if not fetched_articles:
        raise AgentPipelineError("Could not load any Hacker News article details. Try again in a minute.")

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
    
    selected_index = getattr(result, "selected_index", 0)
    try:
        selected_index = int(selected_index)
    except (TypeError, ValueError):
        selected_index = 0
    selected_index = max(0, min(selected_index, len(articles) - 1))

    relevance_score = getattr(result, "relevance_score", 0)
    try:
        relevance_score = max(1, min(int(relevance_score), 10))
    except (TypeError, ValueError):
        relevance_score = 0

    return {
        "selected_article": articles[selected_index],
        "relevance_score": relevance_score,
        "target_audience": getattr(result, "target_audience", "Software Engineers")
    }

def writer_node(state: AgentState) -> dict:
    print("✍️ [Node] Drafting LinkedIn post with Gemini...")
    article = state["selected_article"]
    audience = state["target_audience"]
    
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
    
    draft_post = enforce_article_url(extract_llm_text(result), article["url"])
    return {"draft_post": draft_post}

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
    
    draft_post = enforce_article_url(extract_llm_text(result), article["url"])
    return {"draft_post": draft_post}

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
# 5. REUSABLE API FUNCTIONS (for web frontend)
# ==========================================
def run_agent_pipeline():
    """
    Runs fetch → filter → write pipeline and returns the state dict
    with the first draft ready for human review.
    """
    state = {
        "raw_news": [],
        "selected_article": {},
        "relevance_score": 0,
        "target_audience": "",
        "draft_post": "",
        "feedback": [],
        "is_approved": False
    }

    # Step 1: Fetch news
    result = fetch_news_node(state)
    state.update(result)

    # Step 2: Filter and critique
    result = filter_and_critique_node(state)
    state.update(result)

    # Step 3: Write draft
    result = writer_node(state)
    state.update(result)

    return state


def revise_draft(state, feedback_text):
    """
    Runs the revise node with accumulated feedback and returns updated state.
    """
    # Append new feedback to history
    state["feedback"] = state.get("feedback", []) + [feedback_text]

    result = revise_node(state)
    state.update(result)

    return state


def approve_and_send(state):
    """
    Marks the draft as approved and sends the email.
    Returns a status message.
    """
    state["is_approved"] = True
    send_email_node(state)
    return state


# ==========================================
# 6. TERMINAL ENTRYPOINT
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
