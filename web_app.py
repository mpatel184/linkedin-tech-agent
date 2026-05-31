import os
import sys
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from agent import AgentPipelineError, approve_and_send, revise_draft, run_agent_pipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==========================================
# APP SETUP
# ==========================================
app = FastAPI(title="LinkedIn Tech Agent")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# In-memory session store (single-user local use)
sessions: dict = {}


def error_response(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse({"success": False, "error": message}, status_code=status_code)


def get_or_create_session(request: Request) -> str:
    """Get session ID from cookie or create a new one."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
    return session_id


# ==========================================
# ROUTES
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main frontend page."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/generate")
async def generate_draft(request: Request):
    """
    Run the full pipeline: fetch news → filter → draft.
    Returns the first draft and article metadata.
    """
    try:
        state = run_agent_pipeline()

        session_id = str(uuid.uuid4())
        sessions[session_id] = state

        response = JSONResponse({
            "success": True,
            "draft": state["draft_post"],
            "article": state["selected_article"],
            "relevance_score": state["relevance_score"],
            "target_audience": state["target_audience"],
            "feedback_history": state.get("feedback", []),
            "revision_count": 0
        })
        response.set_cookie("session_id", session_id, httponly=True)
        return response

    except AgentPipelineError as e:
        return error_response(str(e), status_code=503)
    except Exception as e:
        print(f"Unexpected generation error: {e}")
        return error_response("Unexpected server error while generating the draft.", status_code=500)


@app.post("/api/revise")
async def revise(request: Request):
    """
    Accept feedback text and revise the current draft.
    Returns the updated draft.
    """
    session_id = get_or_create_session(request)
    if session_id not in sessions:
        return error_response("No active session. Generate a draft first.", status_code=400)

    body = await request.json()
    feedback_text = body.get("feedback", "").strip()
    if not feedback_text:
        return error_response("Feedback text is required.", status_code=400)

    try:
        state = sessions[session_id]
        state = revise_draft(state, feedback_text)
        sessions[session_id] = state

        response = JSONResponse({
            "success": True,
            "draft": state["draft_post"],
            "article": state["selected_article"],
            "relevance_score": state["relevance_score"],
            "target_audience": state["target_audience"],
            "feedback_history": state.get("feedback", []),
            "revision_count": len(state.get("feedback", []))
        })
        response.set_cookie("session_id", session_id, httponly=True)
        return response

    except AgentPipelineError as e:
        return error_response(str(e), status_code=503)
    except Exception as e:
        print(f"Unexpected revision error: {e}")
        return error_response("Unexpected server error while revising the draft.", status_code=500)


@app.post("/api/approve")
async def approve(request: Request):
    """
    Approve the current draft and send the email notification.
    """
    session_id = get_or_create_session(request)
    if session_id not in sessions:
        return error_response("No active session. Generate a draft first.", status_code=400)

    try:
        state = sessions[session_id]
        approve_and_send(state)

        # Clear the session after approval
        del sessions[session_id]

        return JSONResponse({
            "success": True,
            "message": "Draft approved and email sent successfully!"
        })

    except Exception as e:
        print(f"Unexpected approval error: {e}")
        return error_response("Unexpected server error while approving the draft.", status_code=500)


# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8501, reload=True)
