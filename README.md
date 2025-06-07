# Slack MCP Integration System

A serverless system that connects Slack with MCP (Model Context Protocol) servers to enable AI tool interactions through chat.

## Overview

This project is an AWS Lambda-based system that allows Slack users to interact with various AI tools supporting the MCP protocol through chat conversations. It uses Claude Sonnet 4 as an intermediary to convert natural language requests into appropriate tool calls and provide results in a user-friendly format.

## Key Features

- **Slack Event Processing**: Real-time processing of Slack messages and mention events
- **MCP Server Connection**: Communication with MCP servers via SSE (Server-Sent Events)
- **AI Model Integration**: Integration with Amazon Bedrock's Claude Sonnet 4 model
- **Thread Context Awareness**: Maintains conversation history for contextual responses
- **Real-time Tool Results Display**: Formats tool execution results using Slack Block Kit
- **Asynchronous Processing**: Optimizes response time through gateway-based async message handling

## System Architecture

```
Slack App → API Gateway → slack-mcp-gateway → slack-mcp-client
                                                     ↓
                                              MCP Server (SSE)
                                                     ↓
                                              Various Tools
```

### Components

#### 1. slack_mcp_gateway
- **Role**: Receives Slack events, provides immediate response, then asynchronously invokes client function
- **Features**: 
  - Solves Slack's 3-second response timeout limitation
  - Event relay and load distribution

#### 2. slack_mcp_client
- **Role**: Handles main business logic
- **Key Features**:
  - MCP server and SSE connection management
  - Conversation processing with Claude model
  - Tool execution and result processing
  - Slack message formatting and sending
  - Thread history management

## Installation & Setup

### 1. Environment Variables

The following environment variables must be configured in the AWS Lambda functions:

```bash
# Slack related
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# MCP server connection
MCP_SSE_SERVER_URL=https://your-mcp-server-url

# AWS region (optional)
AWS_REGION=us-east-1
```

### 2. Slack App Configuration

1. Create a new app in the [Slack API Console](https://api.slack.com/apps)
2. Add the following scopes:
   - `app_mentions:read`
   - `channels:history`
   - `chat:write`
   - `im:read`
   - `im:write`
3. Enable Event Subscriptions:
   - `app_mention`
   - `message.im`
4. Set Request URL to your API Gateway endpoint

### 3. AWS Lambda Deployment

```bash
# Grant permissions
chmod +x update_code.sh

# Deploy Lambda functions
./update_code.sh
```

## Dependencies

### Python Packages (requirements.txt)
- `slack_sdk`: Slack API communication
- `requests`: HTTP request handling
- `mcp`: MCP protocol implementation

### AWS Services
- AWS Lambda (function execution)
- Amazon Bedrock (Claude model access)
- API Gateway (HTTP endpoint provision)

## Usage

### Basic Usage

1. Invite the bot to a Slack channel or start a DM conversation
2. Mention the bot or send a DM:
   ```
   @your-bot show me the file list
   ```
3. The bot processes the request using MCP tools and returns results

### Thread Conversations

- Continuing conversations within threads maintains previous context
- The bot analyzes conversation history to use appropriate tools contextually

### Language Support

- Primary support for English
- Automatically adjusts response language to match user query language

## Development & Customization

### Adding New Tools

When new tools are added to the MCP server, Claude will automatically recognize and use them. No client code modifications are required.

### Modifying Message Formatting

Customize how tool results are displayed in Slack by modifying the `_format_tool_result_for_slack` function in `slack_utils.py`.

### Error Handling

The system handles the following error scenarios:
- MCP server connection failures
- Slack API call failures
- Claude model response errors
- Tool execution errors

## Logging & Monitoring

- Detailed execution logs available through AWS CloudWatch Logs
- Tracking of execution time and error status for each step
- Slack message send success/failure logging

## License

Add your project license information here.

## Contributing

How to contribute to the project:
1. Fork and create a feature branch
2. Commit your changes
3. Create a Pull Request

## Troubleshooting

### Common Issues

1. **Bot not responding**
   - Check Slack tokens and permissions
   - Review Lambda function logs
   - Verify MCP server status

2. **Tool execution failures**
   - Verify MCP_SSE_SERVER_URL environment variable
   - Check MCP server connection status

3. **Permission errors**
   - Check AWS IAM role permissions
   - Verify Bedrock model access permissions
