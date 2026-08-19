# Example Notebooks

This directory contains Jupyter notebooks demonstrating usage examples, tutorials, and key features of the deep-research platform.

## Purpose

Use this directory for:
- **Usage Tutorials**: Step-by-step guides for using agents and APIs
- **Feature Demonstrations**: Showcasing key capabilities
- **Onboarding Materials**: Helping new team members get started
- **Reference Implementations**: Example patterns for common tasks

## Guidelines

1. **Naming**: Use clear, descriptive names
   - `agent_name_usage_example.ipynb`
   - `getting_started_tutorial.ipynb`
   - `api_builder_walkthrough.ipynb`

2. **Documentation**: Examples should be well-documented with:
   - Clear purpose statement
   - Prerequisites and setup instructions
   - Explanatory markdown cells throughout
   - Expected outputs

3. **Outputs**: Always commit with outputs so users can see expected results

4. **Maintenance**: Keep examples up-to-date with API changes

5. **Self-Contained**: Examples should run independently with minimal setup

## Suggested Examples to Add

- Agent usage examples (correlation, reimbursement, recommendation agents)
- API builder walkthrough
- Semantic model configuration tutorial
- LangGraph orchestration examples
- Data integration examples (Snowflake, EHAP)

## Running Examples

```bash
# Activate environment
source .venv/bin/activate

# Launch JupyterLab
jupyter lab

# Or run a specific example
jupyter nbconvert --to html --execute notebooks/examples/example_name.ipynb
```
