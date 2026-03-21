I want to build an agentic setup where a central "Director" agent manages a swarm of sub-agents. One agent reads arXiv for new papers, another generates implementation plans, another writes the code, and another reviews it.

You need to use claude code sdk for everthing here. The whole documentaton is here https://platform.claude.com/docs/en/agent-sdk/overview

Write seperate skills for each agent and sub agent. Add user approval and input. Use slash commands if needed https://platform.claude.com/docs/en/agent-sdk/slash-commands

for arxiv agent use the following mcp
{
  "mcpServers": {
    "arxiv-mcp-server": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "ARXIV_STORAGE_PATH",
        "-v",
        "/local-directory:/local-directory",
        "mcp/arxiv-mcp-server"
      ],
      "env": {
        "ARXIV_STORAGE_PATH": "/Users/local-test/papers"
      }
    }
  }
}

Tool: read_paper
Read the full text content of a previously downloaded and converted research paper in clean markdown format. This tool retrieves the complete paper content including abstract, introduction, methodology, results, conclusions, and references. The content is formatted for easy reading and analysis, with preserved mathematical equations and structured sections. Use this tool when you need to access the full text of a paper for detailed study, quotation, analysis, or research. The paper must have been previously downloaded using the download_paper tool.

The papers will be uploaded in an input folder with week_id, always take the latest week_id papers. The paper ids will be an arrray like ['2301.07041', '1706.03762',]. The python agent should spin up a sandbox and execute each code.

The user will provide the objective what he needs to be get built from the papers while starting the agent.