# LinkedIn Tech Agent

An AI-assisted LinkedIn post generator for technical content. The app fetches current Hacker News stories, uses an LLM to pick a relevant engineering topic, drafts a LinkedIn post, lets you revise it from the browser, and can email the approved draft to you.

## Features

- FastAPI web interface at `http://127.0.0.1:8501`
- Hacker News story fetching
- Gemini 2.5 Flash generation with Hugging Face fallback support
- Browser-based draft review and revision loop
- Copy button for the final LinkedIn post
- Optional email delivery after approval
- Terminal LangGraph mode for the original interactive flow

## Project Structure

```text
.
├── agent.py              # LangGraph nodes and reusable pipeline functions
├── web_app.py            # FastAPI app and API routes
├── templates/
│   └── index.html        # Browser UI
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root. You can start from `.env.example`.

```env
GOOGLE_API_KEY=your_gemini_api_key_here
HF_TOKEN=optional_hugging_face_token_here

EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
EMAIL_RECEIVER=receiver_email@gmail.com
```

`GOOGLE_API_KEY` or `HF_TOKEN` is required. Email settings are optional; if they are missing, approval still works but email sending is skipped.

## Run The Web App

Start the FastAPI server:

```bash
python web_app.py
```

Open this URL in your browser:

```text
http://127.0.0.1:8501
```

Use the app:

1. Click `Generate Draft`.
2. Review the selected article and generated LinkedIn post.
3. Enter feedback and click `Revise` if needed.
4. Click `Copy` to copy the post, or `Approve & Send Email` if email variables are configured.

## Run The Terminal Agent

```bash
python agent.py
```

The terminal mode runs the LangGraph workflow, shows the draft in the console, and waits for `yes` or revision feedback.

## Troubleshooting

- If generation says it cannot fetch Hacker News, check your internet connection or proxy settings.
- If generation says Gemini failed, verify `GOOGLE_API_KEY` and quota. Add `HF_TOKEN` if you want the fallback model.
- If email is skipped, create a Gmail App Password and set `EMAIL_SENDER`, `EMAIL_PASSWORD`, and `EMAIL_RECEIVER`.
