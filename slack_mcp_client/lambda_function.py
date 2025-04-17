import json
import logging
import os
import asyncio # Added
from typing import Optional # Added
from contextlib import AsyncExitStack # Added

import boto3
from slack_utils import verify_slack_request, post_message_to_slack, parse_slack_event_body
from mcp import ClientSession # Added
from mcp.client.sse import sse_client # Added

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Bedrock Configuration ---
boto3_session = boto3.Session() # Consider region_name if needed
bedrock_runtime = boto3_session.client('bedrock-runtime')
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# --- MCP Server Configuration ---
# Use MCP_SSE_SERVER_URL for the SSE endpoint
MCP_SSE_SERVER_URL = os.environ.get('MCP_SSE_SERVER_URL')


# --- Add MCPClient Class (adapted from client.py) ---
class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        self.exit_stack = AsyncExitStack() # Use AsyncExitStack for proper context management

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        if not server_url:
            logger.error("MCP_SSE_SERVER_URL environment variable is not set.")
            raise ValueError("MCP_SSE_SERVER_URL is not configured.")

        logger.info(f"Connecting to SSE MCP server at {server_url}")
        try:
            # Use AsyncExitStack to manage contexts
            self._streams_context = await self.exit_stack.enter_async_context(sse_client(url=server_url))
            self._session_context = await self.exit_stack.enter_async_context(ClientSession(*self._streams_context))
            self.session: ClientSession = self._session_context

            # Initialize
            await self.session.initialize()

            # List available tools to verify connection
            logger.info("Initialized SSE client...")
            logger.info("Listing tools...")
            response = await self.session.list_tools()
            tools = response.tools
            logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")

        except Exception as e:
            logger.error(f"Failed to connect or initialize SSE client: {e}", exc_info=True)
            # Ensure cleanup happens even if connection fails partially
            await self.cleanup()
            raise # Re-raise the exception to signal failure

    async def cleanup(self):
        """Properly clean up the session and streams using AsyncExitStack"""
        logger.info("Cleaning up MCPClient resources.")
        await self.exit_stack.aclose()
        self.session = None
        self._streams_context = None
        self._session_context = None


    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools via MCP SSE session"""
        if not self.session:
            logger.error("MCP session is not initialized. Cannot process query.")
            return "Error: Connection to the tool server is not established."

        system_prompt = [
            {
                "text": "You are a helpful assistant integrated with an MCP system. Use the provided tools ONLY when the user's request clearly and explicitly matches a tool's specific purpose described in its description. For general questions, requests for information not covered by the tools, or greetings, answer directly based on your knowledge without attempting to use any tool."
            }
        ]
        messages = [
            {
                "role": "user",
                "content": [{"text": query}]
            }
        ]

        try:
            logger.info("Listing tools from MCP session...")
            response = await self.session.list_tools()
            available_tools = [{
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": tool.inputSchema
                    }
                }
            } for tool in response.tools]
            logger.info(f"Using {len(available_tools)} tools for Bedrock call.")

            # Initial Claude API call
            logger.info("Making initial call to Bedrock converse API.")
            response = bedrock_runtime.converse(
                modelId=CLAUDE_MODEL_ID,
                messages=messages,
                system=system_prompt,
                toolConfig={
                    'tools': available_tools
                }
            )

            # Process response and handle tool calls
            tool_results = []
            final_text = []

            response_message = response['output']['message']
            messages.append(response_message)
            stop_reason = response.get('stopReason')
            logger.debug(f"Bedrock initial response stop_reason: {stop_reason}, message: {response_message}")


            while stop_reason == 'tool_use':
                logger.info("Bedrock requested tool use.")
                tool_use_requests = [
                    content for content in response_message['content'] if content.get('toolUse')]
                tool_result_contents = []

                # Create tasks for concurrent tool execution if needed, or run sequentially
                for tool_request in tool_use_requests:
                    tool_id = tool_request['toolUse']['toolUseId']
                    tool_name = tool_request['toolUse']['name']
                    tool_input = tool_request['toolUse']['input']

                    # Execute tool call via MCP session
                    logger.info(f"--- Calling MCP tool '{tool_name}' with input: {tool_input} ---")
                    result = await self.session.call_tool(tool_name, tool_input)
                    logger.info(f"--- MCP tool '{tool_name}' result: {result} ---")

                    # Append user-facing text (optional, can be noisy in Slack)
                    # final_text.append(f"[Calling tool {tool_name} with args {tool_input}]")

                    # Prepare tool result content for the API
                    tool_output_content = []
                    if result.isError:
                        logger.error(f"Tool execution failed for '{tool_name}': {result.content}")
                        tool_output_content.append({"text": f"Tool execution failed: {result.content}"})
                        # status = 'error' # Set status if needed
                    elif result.content and isinstance(result.content[0], dict) and 'json' in result.content[0]:
                        tool_output_content.append({"json": result.content[0]['json']})
                    elif result.content and hasattr(result.content[0], 'text'):
                         # Ensure result is stringified if Bedrock expects text
                        tool_output_content.append({"text": str(result.content[0].text)})
                    else:
                        logger.warning(f"Tool '{tool_name}' returned unexpected content format: {result.content}")
                        tool_output_content.append({"text": "Tool returned unexpected content format."})

                    tool_result_contents.append({
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": tool_output_content,
                            # "status": status # Add if tracking errors explicitly
                        }
                    })

                # Create the user message containing tool results
                tool_result_message = {
                    "role": "user",
                    "content": tool_result_contents
                }
                messages.append(tool_result_message)

                # Get next response from Claude
                logger.info("--- Sending tool results back to Bedrock ---")
                response = bedrock_runtime.converse(
                    modelId=CLAUDE_MODEL_ID,
                    messages=messages,
                    system=system_prompt, # Re-include system prompt
                    toolConfig={'tools': available_tools}
                )
                response_message = response['output']['message']
                messages.append(response_message)
                stop_reason = response.get('stopReason')
                logger.debug(f"Bedrock response after tool use stop_reason: {stop_reason}, message: {response_message}")


                # Append text content from the new response if any
                assistant_text_content = [
                    c.get('text') for c in response_message.get('content', []) if 'text' in c]
                if assistant_text_content:
                    # Append intermediate text if desired, or just wait for final response
                    # final_text.append("\n".join(assistant_text_content))
                    pass # Often better to just return the final summary


            # Handle final response
            if stop_reason == 'end_turn':
                logger.info("Bedrock conversation finished.")
                assistant_text_content = [
                    c.get('text') for c in response_message.get('content', []) if 'text' in c]
                if assistant_text_content:
                    final_text.append("\n".join(assistant_text_content))
            else:
                # Handle other stop reasons like max_tokens, error, etc.
                logger.warning(f"Bedrock conversation stopped for reason: {stop_reason}. Extracting any available text.")
                assistant_text_content = [
                    c.get('text') for c in response_message.get('content', []) if 'text' in c]
                if assistant_text_content:
                    final_text.append("\n".join(assistant_text_content))
                else:
                    final_text.append("Sorry, the conversation ended unexpectedly.")


            # Join all collected text pieces (if intermediate text was collected)
            # Or just return the last response's text
            final_response_text = "\n".join(final_text)
            if not final_response_text: # Fallback if no text content found
                 logger.error(f"Could not extract final text response from Claude. Last message: {response_message}")
                 return "Sorry, I couldn't generate a response."

            logger.info(f"Final response for user: {final_response_text}")
            return final_response_text

        except Exception as e:
            logger.error(f"Error during Bedrock conversation or MCP tool call: {e}", exc_info=True)
            return "Sorry, there was an error processing your request."


# --- Remove old Bedrock call function ---
# def call_bedrock_with_tool_use(user_message, channel_id, user_id): ...


async def handle_slack_event_async(slack_event, mcp_client: MCPClient):
    """
    Asynchronous handler for Slack events using the MCPClient.
    """
    event_type = slack_event.get('type')
    logger.info(f"Received Slack event type: {event_type}")

    # Process mentions or DMs
    # Ensure the logic here correctly identifies messages intended for the bot
    if event_type == 'app_mention' or (event_type == 'message' and 'subtype' not in slack_event and 'bot_id' not in slack_event):
        user_id = slack_event.get('user')
        text = slack_event.get('text', '').strip()
        channel_id = slack_event.get('channel')
        thread_ts = slack_event.get('ts') # Timestamp for threading replies

        # Basic check to prevent processing empty messages
        if not text:
            logger.info("Ignoring empty message.")
            return

        # More robustly remove the bot mention (assuming it's at the beginning)
        # Replace <@BOT_ID> with actual bot ID if available, otherwise use regex or split
        # This part needs careful testing with actual Slack payloads
        # Example: find the first occurrence of <@...> and take text after it
        mention_end_index = text.find('>')
        if text.startswith('<@') and mention_end_index != -1:
             user_query = text[mention_end_index + 1:].strip()
             if not user_query:
                 logger.info(f"Ignoring message with only bot mention from user {user_id}.")
                 # Optionally send a help message
                 # post_message_to_slack(channel_id, "How can I help you?", thread_ts=thread_ts)
                 return
        else:
            user_query = text # Assume it's a DM or mention was handled differently


        logger.info(f"Processing query from user {user_id} in channel {channel_id}: '{user_query}'")

        # Ensure MCP client is connected before processing
        if not mcp_client.session:
            logger.error("MCP Client session not available. Cannot process query.")
            post_message_to_slack(channel_id, "Sorry, I'm having trouble connecting to my backend services.", thread_ts=thread_ts)
            return

        # Call MCPClient's process_query
        try:
            final_response = await mcp_client.process_query(user_query)
            post_message_to_slack(channel_id, final_response, thread_ts=thread_ts)
        except Exception as e:
            logger.error(f"Error processing query via MCPClient: {e}", exc_info=True)
            post_message_to_slack(channel_id, "Sorry, an unexpected error occurred while processing your request.", thread_ts=thread_ts)

    else:
        logger.debug(f"Ignoring Slack event type: {event_type} or subtype: {slack_event.get('subtype')}")


def lambda_handler(event, context):
    """
    AWS Lambda entry point. Handles Slack verification and routes events.
    Uses asyncio.run to manage the async MCPClient operations.
    """
    logger.debug(f"Received raw event: {json.dumps(event)}")

    # --- Verify Slack Request ---
    if not verify_slack_request(event):
        logger.error("Slack request verification failed.")
        return {'statusCode': 403, 'body': json.dumps({'error': 'Verification failed'})}
    logger.info("Slack request verification passed.")

    # --- Parse Slack Event Body ---
    body = parse_slack_event_body(event)
    if body is None:
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid JSON body'})}

    # --- Slack URL Verification Challenge ---
    if body.get('type') == 'url_verification':
        challenge = body.get('challenge')
        logger.info(f"Responding to Slack URL verification challenge: {challenge}")
        return {'statusCode': 200, 'headers': {'Content-Type': 'application/json'}, 'body': json.dumps({'challenge': challenge})}

    # --- Handle Slack Callback Events Asynchronously ---
    if body.get('type') == 'event_callback':
        slack_event = body.get('event', {})

        # Acknowledge Slack immediately
        ack_response = {'statusCode': 200, 'body': json.dumps({'message': 'Event received, processing asynchronously'})}

        # Run the async event handling logic
        async def run_async_handler():
            mcp_client = MCPClient()
            try:
                await mcp_client.connect_to_sse_server(server_url=MCP_SSE_SERVER_URL)
                await handle_slack_event_async(slack_event, mcp_client)
            except ValueError as ve: # Catch configuration error specifically
                 logger.error(f"Configuration error: {ve}")
                 # Optionally notify Slack if possible/needed, though ack already sent
            except Exception as e:
                logger.error(f"Error in async handler: {e}", exc_info=True)
                # Attempt to notify Slack about the error if channel_id is accessible
                channel_id = slack_event.get('channel')
                thread_ts = slack_event.get('ts')
                if channel_id:
                     try:
                         post_message_to_slack(channel_id, "Sorry, an internal error occurred while processing your request.", thread_ts=thread_ts)
                     except Exception as post_error:
                          logger.error(f"Failed to post error message to Slack: {post_error}")
            finally:
                await mcp_client.cleanup()

        # Use asyncio.create_task if running in an environment where the loop persists briefly after response
        # Or if using Lambda extensions/async frameworks. For standard Lambda, run might block ack.
        # A better approach for Lambdas needing > 3s is SQS queueing.
        # For simplicity here, we'll run it and rely on Lambda execution context.
        # WARNING: If connect/process takes >3s, Slack might retry, leading to duplicate processing.
        # Consider adding message deduplication (e.g., using event ID) or using SQS.

        try:
             asyncio.run(run_async_handler())
             logger.info("Async handler processing initiated.")
        except Exception as e:
             # Catch errors during the setup/run of asyncio itself
             logger.error(f"Failed to run async handler: {e}", exc_info=True)
             # Return 500 only if the sync part fails critically before ack
             # ack_response = {'statusCode': 500, 'body': json.dumps({'error': 'Failed to start processing'})}


        return ack_response # Return 200 OK quickly

    else:
        logger.warning(f"Received unhandled callback type: {body.get('type')}")
        return {'statusCode': 200, 'body': json.dumps({'message': 'Unhandled event type received'})}

# Note: Need to add 'mcp-client-library' (or the actual name) and 'asyncio' (if not built-in)
# to requirements.txt and ensure the Lambda environment has these packages.
# Also ensure SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, and MCP_SSE_SERVER_URL are set as environment variables.
