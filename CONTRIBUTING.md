# Contributing to CNC Copilot

Thank you for your interest in contributing! We welcome community involvement to make manufacturing smarter and safer.

## Getting Started

1.  **Fork** the repository.
2.  **Clone** your fork:
    ```bash
    git clone https://github.com/your-username/Dev-contitional.git
    cd Dev-contitional
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Environment**:
    ```bash
    cp .env.example .env
    ```

## Development Workflow

-   **Backend**: The core logic resides in `advanced_cnc_copilot/backend`.
-   **Frontend**: The UI is in `advanced_cnc_copilot/frontend`.
-   **Mock Mode**: By default, the system runs in MOCK mode (Mock CNC, Mock Kafka, Mock LLM). This allows you to develop without hardware.

## Running the App

```bash
python3 advanced_cnc_copilot/backend/main.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

## Submitting Changes

1.  Create a new branch: `git checkout -b feature/my-new-feature`
2.  Commit your changes: `git commit -m 'Add new feature'`
3.  Push to the branch: `git push origin feature/my-new-feature`
4.  Submit a **Pull Request**.

## Code Style

-   Follow PEP 8 for Python.
-   Ensure new features have corresponding verification scripts or tests.
