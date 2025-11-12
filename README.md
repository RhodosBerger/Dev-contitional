# Dev-conditional

**Autonomous Server Application Engine with LLM Integration**

A powerful platform for creating, managing, and executing autonomous server applications through visual workflow design, AI-powered assistance, and automated code generation.

![Dev-conditional Logo](https://via.placeholder.com/400x200/667eea/ffffff?text=Dev-conditional)

## 🚀 Features

### 🎨 Visual Workflow Designer
- **Node-based Editor**: Create complex automation workflows with an intuitive drag-and-drop interface
- **Conditional Logic**: Implement sophisticated business rules with if/else conditions, switches, and validators
- **Real-time Collaboration**: Work together with team members on workflow design
- **Template Library**: Pre-built workflow templates for common use cases

### 🤖 AI-Powered Assistance
- **LLM Integration**: Get intelligent suggestions for workflow design and optimization
- **Code Generation**: Generate production-ready code from workflow configurations
- **Best Practice Recommendations**: AI-powered suggestions for security, performance, and maintainability
- **Natural Language Interface**: Describe your goals and let AI suggest the optimal workflow structure

### ⚡ Advanced Code Generation
- **Multiple Framework Support**: Generate applications using FastAPI, Express.js, and more
- **Template System**: Customizable templates for different project types and architectures
- **Database Integration**: Automatic ORM setup, migrations, and CRUD operations
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

### 🔧 Autonomous Execution
- **Terminal Agents**: Secure execution environment for generated code
- **Error Handling**: Comprehensive error recovery and retry mechanisms
- **Monitoring**: Real-time execution tracking and performance metrics
- **Scalability**: Kubernetes-ready deployment configurations

### 📊 Real-time Features
- **WebSocket Communication**: Live updates for workflow execution and AI assistance
- **Event Streaming**: Kafka-based message queuing for scalable event handling
- **Monitoring Dashboard**: Track system health, performance, and usage metrics

## 🏗️ Architecture

```
Dev-conditional/
├── backend/                 # Core server engine
│   ├── src/
│   │   ├── api/            # REST API endpoints
│   │   ├── codegen/        # Code generation engine
│   │   ├── llm/           # LLM integration service
│   │   ├── storage/       # Database models and configuration
│   │   ├── websocket/     # Real-time communication
│   │   └── workflow/      # Workflow execution engine
│   └── tests/
├── frontend/               # Web interface
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── editor/        # Visual workflow editor
│   │   └── dashboard/     # Admin interface
├── templates/             # Code generation templates
├── docker/               # Container configurations
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL (optional, can use SQLite for development)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/dev-contitional.git
cd dev-contitional
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install
```

4. **Start the services**
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or manually:
# Start backend
cd backend && uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in new terminal)
cd frontend && npm run dev
```

5. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📚 Usage

### Creating Your First Workflow

1. **Open the Workflow Designer**
   - Navigate to http://localhost:3000
   - Click "Create Workflow"

2. **Add Nodes**
   - **Trigger Node**: Start your workflow (manual, webhook, schedule)
   - **API Call Node**: Make HTTP requests to external services
   - **Condition Node**: Add branching logic
   - **LLM Prompt Node**: Get AI assistance for decisions
   - **Code Execution Node**: Run custom code safely

3. **Connect Nodes**
   - Drag connections between nodes to define data flow
   - Configure node parameters in the properties panel

4. **Test and Execute**
   - Click "Test" to validate your workflow
   - Click "Execute" to run the workflow with sample data

### Generating Code from Workflows

1. **Configure Generation Settings**
   - Select your target framework (FastAPI, Express.js, etc.)
   - Choose database type and authentication method
   - Set project name and description

2. **Generate Project**
   - Click "Generate Code" to create a full application
   - Preview the generated files and structure
   - Download as ZIP or deploy directly

3. **Deploy Generated Application**
   - Use the provided Dockerfile for containerization
   - Follow the deployment guide in the generated README

### Using AI Assistance

1. **Chat Interface**
   - Access the AI assistant from any workflow screen
   - Ask questions about best practices, architecture, or debugging
   - Get suggestions for workflow improvements

2. **Code Review**
   - Use AI to validate generated code
   - Get security and performance recommendations
   - Refactor code for better maintainability

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/devcontitional

# LLM Integration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Security
SECRET_KEY=your-secret-key-here

# Kafka (for production)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Database Setup

**PostgreSQL (Production)**
```sql
CREATE DATABASE devcontitional;
CREATE USER devcontitional WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE devcontitional TO devcontitional;
```

**SQLite (Development)**
```env
DATABASE_URL=sqlite:///./devcontitional.db
```

## 📖 API Documentation

### Core Endpoints

- `GET /health` - System health check
- `POST /api/workflow/` - Create new workflow
- `GET /api/workflow/` - List all workflows
- `POST /api/codegen/generate` - Generate code from workflow
- `POST /api/llm/chat` - Chat with AI assistant

### WebSocket Connections

- `/ws/chat/{session_id}` - Real-time AI chat
- `/ws/workflow/{workflow_id}` - Workflow execution updates

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Integration Tests
```bash
# Run full test suite
npm run test

# With coverage
npm run test:coverage
```

## 🚀 Deployment

### Docker Deployment

1. **Build Images**
```bash
docker build -f docker/Dockerfile.backend -t devcontitional-backend .
docker build -f docker/Dockerfile.frontend -t devcontitional-frontend .
```

2. **Deploy with Docker Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

### Environment-Specific Configurations

- **Development**: SQLite, debug mode enabled
- **Staging**: PostgreSQL, monitoring enabled
- **Production**: PostgreSQL cluster, Redis cluster, Kafka

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style

- Python: Follow PEP 8, use Black for formatting
- JavaScript/TypeScript: Use ESLint and Prettier
- All commits should follow Conventional Commits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for building APIs
- [React Flow](https://reactflow.dev/) - Highly customizable library for building node-based UIs
- [OpenAI](https://openai.com/) - AI model integration
- [Docker](https://www.docker.com/) - Container platform

## 📞 Support

- 📧 Email: support@devcontitional.com
- 💬 Discord: [Join our community](https://discord.gg/devcontitional)
- 📖 Documentation: [docs.devcontitional.com](https://docs.devcontitional.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/dev-contitional/issues)

## 🗺️ Roadmap

### Version 1.1 (Next Release)
- [ ] Enhanced workflow templates
- [ ] Multi-cloud deployment support
- [ ] Advanced monitoring and alerting
- [ ] Plugin system for custom nodes

### Version 2.0 (Future)
- [ ] Mobile application
- [ ] Advanced AI agents
- [ ] Enterprise SSO integration
- [ ] Advanced security features

---

**Built with ❤️ by the Dev-conditional Team**