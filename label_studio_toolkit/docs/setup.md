# Setup

## 1. Install Python dependencies

The toolkit dependencies (`label-studio-sdk`, `langchain-openai`, etc.) are installed when you install the project from the repo root:

```bash
pip install -e .
```

See the main [README](../../README.md) for full installation options (UV, Make).

## 2. Get a Label Studio instance

You can use a self-hosted instance or the hosted cloud version.

### Option A — Cloud (no install)

Sign up at [https://app.heartex.com/](https://app.heartex.com/) (Label Studio Enterprise / HumanSignal Cloud). Use the URL of your workspace as `LABEL_STUDIO_BASE_URL`.

### Option B — Self-host with Docker

```bash
docker run -it -p 8080:8080 -v $(pwd)/ls-data:/label-studio/data heartexlabs/label-studio:latest
```

Open `http://localhost:8080`, create an account, and you're done. Use `http://localhost:8080` as `LABEL_STUDIO_BASE_URL`.

For production setups (Postgres, persistent volumes, reverse proxy) see the [official Label Studio install docs](https://labelstud.io/guide/install).

## 3. Get a Label Studio API key

1. Open Label Studio in the browser and sign in.
2. Click your account avatar → **Account & Settings**.
3. Copy the **Access Token** under *Personal Access Token*.

## 4. Configure environment variables

Create a `.env` file at the repo root (it is loaded automatically by every script via `python-dotenv`):

```bash
LABEL_STUDIO_BASE_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=<your-label-studio-token>
OPENAI_API_KEY=<your-openai-api-key>
```

`OPENAI_API_KEY` is required by [`LangChainOpenAIAnnotator`](../annotator.py) for any preannotation step.

## 5. Verify

Run a quick sanity check from the repo root:

```bash
python -c "from label_studio_toolkit.api.client import LabelStudioClient; \
import os; \
c = LabelStudioClient(os.environ['LABEL_STUDIO_BASE_URL'], os.environ['LABEL_STUDIO_API_KEY'], 'sanity-check'); \
print('OK, project id =', c.project.id)"
```

This creates an empty project named `sanity-check` in your Label Studio instance. Delete it from the UI when you're done.

## Next

- [workflows.md](workflows.md) — pick the right workflow for your task.
