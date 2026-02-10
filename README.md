# CNC Copilot: Advanced Manufacturing AI

**Bridging the gap between generative AI and industrial CNC machining.**

CNC Copilot is an intelligent assistant designed to help machinists, engineers, and operators optimize their workflows. It integrates:
- **Local LLMs (OpenLLaMA)** for secure, offline code generation.
- **RAG (Retrieval Augmented Generation)** for querying machine logs and manuals.
- **Fanuc FOCAS Integration** (Simulated & Real) for machine telemetry.
- **Modern Web UI** for interaction.

## 🚀 Features

- **G-Code Generation**: Explain, optimize, and generate G-Code.
- **Real-time Telemetry**: Monitor machine status (Run, Stop, Alarm) and load.
- **Log Analysis**: diagnose issues using historical data.
- **Zero-Dependency Fallback**: Runs entirely in Python without heavy external databases if needed.

## 📦 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Dev-contitional.git
    cd Dev-contitional
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env to match your setup (Mock vs Real)
    ```

## 🛠 Usage

### Start the System
Run the main backend entry point. This serves both the **API** and the **Web UI**.

```bash
python3 advanced_cnc_copilot/backend/main.py
```

-   **Frontend**: [http://localhost:8000](http://localhost:8000)
-   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Docker Deployment
For a production-like environment with Redis:

```bash
docker-compose up --build
```

## 🧠 Architecture

-   **Backend**: FastAPI (Python)
-   **Frontend**: Vanilla JS + CSS (Single Page App)
-   **LLM Engine**: Llama.cpp (via Python bindings) or OpenAI/Mock.
-   **Vector Store**: NumPy-based in-memory store (Fallbacks to ChromaDB).

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
