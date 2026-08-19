# Notebooks

This directory contains Jupyter notebooks for exploration, prototyping, and examples.

## Directory Structure

- **prototyping/** - Feature prototyping, proof-of-concepts, and experimental work
- **examples/** - Usage examples, tutorials, and demonstrations

## Guidelines

### General Best Practices

1. **Naming Convention**: Use descriptive names with dates for exploratory work
   - Example: `2026-04-30_hru_graph_analysis.ipynb`
   - Example: `correlation_agent_prototype.ipynb`

2. **Documentation**: Include markdown cells explaining:
   - Purpose of the notebook
   - Prerequisites and setup
   - Key findings or outcomes

3. **Dependencies**: Document any additional dependencies at the top of the notebook
   - Use inline script dependencies with `uv run` when possible

4. **Clean Outputs**: Consider clearing outputs before committing to reduce file size
   - Exception: Keep outputs for example notebooks

### Prototyping Directory

Use this for:
- Testing new features before implementation
- Proof-of-concept work
- Algorithm experimentation
- Data exploration for new agents

### Examples Directory

Use this for:
- Usage tutorials for agents and APIs
- Demonstrations of key features
- Onboarding materials for new team members
- Reference implementations

## Running Notebooks

### Option 1: JupyterLab (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install JupyterLab if not already installed
uv pip install jupyterlab

# Launch JupyterLab
jupyter lab
```

### Option 2: VS Code

1. Open the notebook file in VS Code
2. Select the Python interpreter from `.venv`
3. Run cells interactively

### Option 3: Command Line

```bash
# Activate virtual environment
source .venv/bin/activate

# Run notebook and convert to HTML
jupyter nbconvert --to html --execute notebooks/examples/my_notebook.ipynb
```

## Environment Setup

Notebooks have access to all packages in the workspace:
- `deep_research_core` - Core libraries and agents
- `deep_research_utils` - Utility functions
- `deep_research_agents` - Agent implementations

Make sure to activate the virtual environment before running notebooks:

```bash
source .venv/bin/activate
```

## Contributing

When adding notebooks:
1. Choose the appropriate directory (prototyping vs examples)
2. Use descriptive names
3. Include documentation in markdown cells
4. Test that the notebook runs from top to bottom
5. Consider adding to `.gitignore` if containing sensitive data
