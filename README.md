# LangChain Multiagent System

This project implements a multiagent system using LangChain, demonstrating how to create and use AI agents with various capabilities.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

The current implementation includes a basic agent with a calculator tool. To run the example:

```bash
python agent.py
```

## Features

- Asynchronous agent execution
- Modular tool system
- Easy to extend with new capabilities
- Built with LangChain's latest features

## Extending the System

To add new tools to the agent, extend the `tools` list in the `Agent` class. Each tool should be implemented as a method and added to the tools list using the `Tool` class. 