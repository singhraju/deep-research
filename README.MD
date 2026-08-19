# Deep Research - Healthcare Analytics Investigation Platform

> **Version 2.0.4** - AI-powered healthcare cost analytics and root cause investigation

## Overview

Deep Research is a production-grade healthcare analytics platform that combines LangGraph orchestration with Streamlit UI to investigate cost trends, detect anomalies, and provide evidence-based insights for healthcare analytics teams.

### Key Features

- 🔍 **Intelligent Investigation** - LangGraph-powered orchestration for multi-step analysis
- 📊 **Interactive UI** - Streamlit-based interface with real-time progress tracking
- 🏥 **Healthcare Focus** - Specialized for cost of care, PMPM trends, and claims analysis
- 🔗 **Data Integration** - Snowflake for analytics, EHAP for LLM access
- 🎯 **Root Cause Analysis** - Automated hypothesis generation and evidence correlation
- 📈 **Semantic Modeling** - YAML-based semantic layer for healthcare dimensions

## Quick Start

### Prerequisites

- Python 3.13+ (per `pyproject.toml`)
- UV package manager
- Access to EHAP (Elevance Health API Platform)
- Snowflake credentials

### Installation

```bash
# Clone repository
git clone https://AM67979@bitbucket.elevancehealth.com/scm/idisc/idiscovery-deep-research.git
cd idiscovery-deep-research

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Configure environment
cp .env.sample .env
# Edit .env with your credentials

# Run application
> streamlit run apps/ui/src/app.py
or
> streamlit run apps/ui/src/app.py  --server.headless=true

```

Access the application at `http://localhost:8501`

## Project Structure

```
deep-research/
├── apps/
│   └── ui/                    # Streamlit UI application
│       ├── src/
│       │   ├── app.py        # Main application entry point
│       │   └── ui/           # UI components
│       └── correlation_runs/ # Analysis output
├── packages/
│   ├── core/                 # Core orchestration logic
│   │   └── src/deep_research_core/
│   └── utils/                # Utility libraries
│       └── src/deep_research_utils/
├── configs/                  # Semantic model configurations
│   ├── ecap_semantic_view.yaml
│   └── ecap_semantic_view_with_samples.yaml
├── docs/                     # Architecture documentation
│   └── design/
├── tests/                    # Test suites
├── .github/workflows/        # CI/CD pipelines
├── Dockerfile               # Container configuration
├── pyproject.toml           # Project dependencies
└── version.py               # Version tracking

```

## Architecture

The application follows a layered architecture:

### Control Plane
- **Intent Resolver** - Understands user questions and extracts context
- **LangGraph Orchestrator** - Compiles and supervises investigation workflows

### Execution Plane
- **Stateless Agents** - Data retrieval, anomaly detection, analysis, visualization
- **Evidence Correlation** - Links findings to policies, codes, and trends

### Data Plane
- **Snowflake** - Healthcare analytics data (claims, costs, membership)
- **Semantic Model** - YAML-based dimension and metric definitions

See [docs/design/](./docs/design/) for detailed architecture documentation.

## Usage

### Basic Investigation Flow

1. **Ask a Question**
   ```
   "What caused the spike in California PMPM in Q3 2024?"
   ```

2. **Review Progress**
   - Watch real-time orchestrator steps
   - View LangGraph visualization
   - Monitor agent execution

3. **Analyze Results**
   - Trend visualizations
   - Outlier code analysis
   - Policy correlations
   - Executive summary

### Filters

The UI provides filters for:
- **State** - Geographic dimension
- **Amount Type** - Allowed, paid, incurred
- **Period** - Time window for analysis
- **Incurred Month** - Specific month focus

## Development

### Workspace Structure

This is a UV workspace with multiple packages:

```bash
# Install all packages
uv sync

# Run tests
uv run pytest tests/

# Add dependency to root
uv add <package>

# Add dependency to specific package
uv add --package deep-research-core <package>
```

### Running Tests

```bash
# Unit tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=packages --cov-report=html

# Integration tests
uv run pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code (if ruff configured)
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking (if mypy configured)
uv run mypy packages/
```

## FastAPI server:
> uvicorn packages.server.app:app --reload --host 0.0.0.0 --port 8000

### Docker

```bash
# Build image
docker build -t deep-research:2.0.4 .

# Run container
docker run -p 8501:8501 --env-file .env deep-research:2.0.4
```

### CI/CD

The repository includes GitHub Actions workflows:

- **ci.yml** - Continuous integration (lint, validate, build test)
- **tests.yml** - Test suite execution
- **release.yml** - Build and push to Quay registry

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

### Container Registry

Images are published to:
```
quay-nonprod.elevancehealth.com/programintegrity/idiscovery-deep-research:2.0.4
```

## Configuration

### Environment Variables

Key configuration (see `.env.sample` for full list):

```bash
# LLM Access
EHAP_BASE_URL=https://your-endpoint
EHAP_CLIENT_ID=your_client_id
EHAP_CLIENT_SECRET=your_secret

# Snowflake
SNOWFLAKE_ACCOUNT=account.region
SNOWFLAKE_USER=username
SNOWFLAKE_SECRET=password
SNOWFLAKE_WAREHOUSE=warehouse_name

# Feature Flags
ENABLE_LLM=true
ENABLE_SNOWFLAKE=true
ENABLE_CORRELATION=true
```

### Semantic Model

Configure healthcare dimensions and metrics in:
```
configs/ecap_semantic_view_with_samples.yaml
```

## Documentation

- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Deployment and operations guide
- **[AGENTS.md](./AGENTS.md)** - AI agent development guidelines
- **[docs/design/](./docs/design/)** - Architecture and design documents
- **[packages/core/README.md](./packages/core/README.md)** - Core package documentation
- **[packages/utils/README.md](./packages/utils/README.md)** - Utils package documentation

## Contributing

1. Create feature branch: `git checkout -b feature/PICOCAI-XXXX-description`
2. Make changes and test locally
3. Run tests: `uv run pytest tests/`
4. Commit with conventional commits
5. Push and create pull request

## Version History

- **2.0.4** (Current) - LangGraph orchestration release line (2.0.x)
- **1.0.7** - Previous POC version (deprecated)

## Support

For questions or issues:
- Review documentation in `docs/`
- Check [DEPLOYMENT.md](./DEPLOYMENT.md) troubleshooting section
- Contact the development team
- See [AGENTS.md](./AGENTS.md) for AI agent guidance

## License

Copyright 2026 Elevance Health. All rights reserved.

---

**Built with** ❤️ **by the Elevance Health Program Integrity Team**