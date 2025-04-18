import json
import logging
import os
import asyncio # Added
from typing import Optional, Any, Dict # Added
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

    def _format_tool_result_for_slack(self, tool_name: str, result_data: Any, is_error: bool = False) -> Dict[str, Any]:
        """Formats tool results generically into Slack Block Kit or plain text."""
        blocks = []
        fallback_text = f"Result from tool: {tool_name}" # Default fallback text

        if is_error:
            error_text = str(result_data)
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f":x: *Error executing {tool_name}*:\n```\n{error_text}\n```"}
            })
            fallback_text = f"Error executing {tool_name}: {error_text}"
            return {"blocks": blocks, "text": fallback_text}

        # --- Generic Dictionary (JSON) Formatting ---
        if isinstance(result_data, dict):
            if not result_data:
                 blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*[Tool: {tool_name}]* Received empty dictionary."}})
                 return {"blocks": blocks, "text": f"Tool {tool_name}: Empty dictionary"} 
                 
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*[Tool: {tool_name}]*"}})
            fallback_parts = [f"Tool {tool_name}:"]

            for key, value in result_data.items():
                if isinstance(value, list) and not value:
                    continue # Skip empty lists

                blocks.append({"type": "divider"})
                key_text = f"*{key.replace('_', ' ').title()}*"
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": key_text}})
                fallback_parts.append(f"{key.replace('_', ' ').title()}:")

                # 1. List of Dictionaries (Generic)
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                     for i, item_dict in enumerate(value):
                         # Combine key info into a single markdown string generically, showing all keys
                         info_parts = []
                         item_fallback = []
                         # Removed keys_processed counter and limit
                         for item_key, item_value in item_dict.items():
                             val_str = str(item_value)
                             if len(val_str) > 50: val_str = val_str[:47] + "..." # Keep value truncation
                             # Format as Key: Value
                             part = f"*{item_key.replace('_',' ').title()}:* {val_str}"
                             info_parts.append(part)
                             item_fallback.append(part) # Add to fallback too
                         
                         # Removed the logic for adding '...'

                         if info_parts:
                            info_string = ' | '.join(info_parts)
                            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": info_string}})
                            # Add to fallback text (might get long, handled later)
                            fallback_parts.append(f"- Item {i+1}: { ', '.join(item_fallback)}")
                         else: # Should not happen if list is not empty, but good practice
                            blocks.append({"type": "context", "elements": [{"type": "plain_text", "text": f" (Item {i+1} - Empty object?)"}]})
                            fallback_parts.append(f"- Item {i+1}: (Empty object?)")
                         # Removed the specific key referencing logic

                # 2. Single Dictionary (Already generic)
                elif isinstance(value, dict):
                    # Combine key info into a single markdown string
                    info_parts = []
                    item_fallback = [] # For fallback text
                    for item_key, item_value in value.items():
                         # Limit the number of items shown in the line
                         if len(info_parts) < 5: 
                             val_str = str(item_value)
                             if len(val_str) > 50: val_str = val_str[:47] + "..."
                             part = f"*{item_key.replace('_',' ').title()}:* {val_str}"
                             info_parts.append(part)
                             item_fallback.append(part)
                             
                    if info_parts:
                         info_string = ' | '.join(info_parts)
                         blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": info_string}})
                         fallback_parts.append(f"Details: { ', '.join(item_fallback)}")
                    else:
                         blocks.append({"type": "context", "elements": [{"type": "plain_text", "text": " (Empty object)"}]})
                         fallback_parts.append("(Empty object)")
                    # Removed fields-based section

                # 3. Simple List
                elif isinstance(value, list):
                     list_items = []
                     for item in value:
                         item_str = str(item)
                         if len(item_str) > 100: item_str = item_str[:100] + "..."
                         list_items.append(f"- {item_str}")
                     list_text = "\n".join(list_items)
                     blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": list_text}})
                     fallback_parts.append(list_text)
                
                # 4. Other Types (String, Number, etc.) or fallback
                else:
                     try:
                         value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                         if len(value_str) > 500: value_str = value_str[:500] + "..."
                         if isinstance(value, (dict, list)) or '\n' in value_str or len(value_str) > 60:
                              blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"```\n{value_str}\n```"}})
                         else:
                              blocks.append({"type": "section", "text": {"type": "plain_text", "text": value_str, "emoji": True}})
                         fallback_parts.append(value_str)
                     except Exception:
                          value_str = str(value)
                          if len(value_str) > 500: value_str = value_str[:500] + "..."
                          blocks.append({"type": "section", "text": {"type": "plain_text", "text": value_str, "emoji": True}})
                          fallback_parts.append(value_str)

            fallback_text = "\n".join(fallback_parts)
            if len(fallback_text) > 300: fallback_text = fallback_text[:300] + "..."
            return {"blocks": blocks, "text": fallback_text}

        # --- Generic String Formatting ---
        elif isinstance(result_data, str):
             fallback_text = f"Tool {tool_name}: {result_data}"
             # Put in code block if multi-line or looks like code
             if '\n' in result_data or result_data.strip().startswith(('{', '[', '<')):
                 blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*[Tool: {tool_name}]*\n```\n{result_data}\n```"}})
             else:
                 blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*[Tool: {tool_name}]*\n{result_data}"}})
             if len(fallback_text) > 300: fallback_text = fallback_text[:300] + "..."
             return {"blocks": blocks, "text": fallback_text}

        # --- Fallback for other types ---
        else:
            fallback_text = f"Tool {tool_name} returned a non-standard result type: {type(result_data)}"
            try:
                result_repr = repr(result_data)
                if len(result_repr) > 500: result_repr = result_repr[:500] + "..."
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*[Tool: {tool_name}]* Received non-standard result:\n```\n{result_repr}\n```"}})
                fallback_text = f"Tool {tool_name}: {result_repr}"
                return {"blocks": blocks, "text": fallback_text}
            except Exception:
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*[Tool: {tool_name}]* Received non-standard and non-representable result."}})
                return {"blocks": blocks, "text": fallback_text}

    async def process_query(self, query: str, channel_id: str, thread_ts: str) -> str:
        """Process a query using Claude and available tools via MCP SSE session,
        posting intermediate tool results to Slack.
        """
        if not self.session:
            logger.error("MCP session is not initialized. Cannot process query.")
            return "Error: Connection to the tool server is not established."

        system_prompt = [
            {
                # Even stricter prompt focusing on factual reporting of tool actions
                "text": "You are a helpful assistant integrated with an MCP system. Use the provided tools ONLY when the user's request clearly and explicitly matches a tool's specific purpose described in its description. For general questions, requests for information not covered by the tools, or greetings, answer directly based on your knowledge without attempting to use any tool.\n\n" 
                        "**When generating your final response after using tools:** Your response MUST be based *strictly* on the results provided back to you in the `toolResult` messages. State ONLY the facts about which tools were called and what actions were confirmed by their results (e.g., which specific instances were modified). DO NOT include general advice, potential benefits (like cost savings), or any information not directly present in the `toolResult`. Avoid introductory phrases. Focus on reporting the factual outcome of the tool executions."
            }
        ]
        messages = [
            {
                "role": "user",
                "content": [{"text": query}]
            }
        ]
        
        initial_user_message = messages[0] # Restore storing the first user message

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
                    logger.debug(f"--- Raw MCP tool '{tool_name}' result object: {result} ---")
                    
                    # --- Handle Tool Result Processing --- 
                    bedrock_content_for_next_step = None 
                    slack_payload = None
                    # Content to be potentially processed by LLM for Slack display
                    content_for_slack_processing = None 
                    is_error_result = False
                    is_json_result = False # Flag to differentiate JSON vs Text for LLM request

                    if result.isError:
                        is_error_result = True
                        logger.error(f"Tool execution failed for '{tool_name}': {result.content}")
                        error_message = str(result.content) if result.content else "Unknown tool execution error"
                        content_for_slack_processing = error_message # Pass error message for formatting
                        bedrock_content_for_next_step = [{"text": f"Tool execution failed: {error_message}"}]
                    
                    elif result.content and isinstance(result.content, list) and len(result.content) > 0:
                        first_content_item = result.content[0]
                        if hasattr(first_content_item, 'text') and isinstance(first_content_item.text, str):
                            potential_json_string = first_content_item.text
                            try:
                                parsed_json = json.loads(potential_json_string)
                                is_json_result = True
                                content_for_slack_processing = parsed_json # Pass parsed dict for LLM summary
                                logger.info(f"--- Parsed JSON from TextContent for tool '{tool_name}' ---")
                                bedrock_content_for_next_step = [{"json": parsed_json}]
                            except json.JSONDecodeError:
                                extracted_text = potential_json_string
                                content_for_slack_processing = extracted_text # Pass raw text for LLM refinement
                                logger.info(f"--- Extracted plain text content from tool '{tool_name}' ---")
                                bedrock_content_for_next_step = [{"text": extracted_text}]
                        else:
                             is_error_result = True # Treat unexpected format as error for display
                             error_text = "Tool returned content in an unexpected list format."
                             content_for_slack_processing = error_text
                             logger.warning(f"Tool '{tool_name}' returned list content with unexpected item type: {type(first_content_item)}")
                             bedrock_content_for_next_step = [{"text": error_text}]
                    else:
                        is_error_result = True # Treat unexpected format as error for display
                        error_text = "Tool returned content in an unexpected format or no content."
                        content_for_slack_processing = error_text
                        logger.warning(f"Tool '{tool_name}' returned unexpected content format or empty content: {result.content}")
                        bedrock_content_for_next_step = [{"text": error_text}]

                    # --- LLM Processing for Slack Display (Refinement for Text ONLY) ---                    
                    slack_display_content = None
                    # Only process non-error, non-JSON string results with LLM
                    if not is_error_result and not is_json_result and isinstance(content_for_slack_processing, str):
                        logger.info(f"--- Requesting LLM refinement/translation for '{tool_name}' text result ---")
                        refinement_request_content = [
                            # Even stricter prompt focusing on exact meaning preservation
                            {"text": f"The user asked the initial query provided earlier. The tool '{tool_name}' produced the text output below. Your task is ONLY to translate/rephrase the EXACT meaning of the provided text output into the same language as the initial user query. Preserve ALL details and words from the original output, including technical terms or specific actions like 'simulated'. DO NOT add any extra words, phrases, introductions, confirmations, or conversational text. Output ONLY the direct translation/rephrasing of the provided text."},
                            {"text": f"Tool Output Text:\n```\n{content_for_slack_processing}\n```"}
                        ]
                        refinement_messages = [ initial_user_message, {"role": "user", "content": refinement_request_content} ]
                        try:
                            refinement_response = bedrock_runtime.converse(modelId=CLAUDE_MODEL_ID, messages=refinement_messages, system=system_prompt)
                            refinement_output = refinement_response.get('output', {}).get('message', {}).get('content', [])
                            slack_display_content = "".join([c.get('text', '') for c in refinement_output if 'text' in c]).strip()
                            if not slack_display_content:
                                 logger.warning(f"LLM refinement was empty for {tool_name}. Using original text.")
                                 # Fallback to original text if LLM returns empty
                                 slack_display_content = content_for_slack_processing 
                        except Exception as refine_err:
                            logger.error(f"Failed to get LLM refinement for '{tool_name}': {refine_err}", exc_info=True)
                            # Fallback to original text on error
                            slack_display_content = content_for_slack_processing 
                            
                    # If not processed by LLM (JSON, error, or LLM failed), use the original content
                    if slack_display_content is None:
                         slack_display_content = content_for_slack_processing
                                   
                    # --- Format final Slack content using the potentially refined display content ---
                    # Note: _format_tool_result_for_slack handles dict (JSON) or string formatting
                    if slack_display_content is not None:
                         slack_payload = self._format_tool_result_for_slack(tool_name, slack_display_content, is_error_result)
                    else:
                         # Should ideally not happen, but as a safety net
                         logger.error(f"slack_display_content was None for tool {tool_name}. Cannot format message.")
                         slack_payload = {"text": f"Error displaying result for tool {tool_name}"}

                    # --- Post Formatted Message to Slack --- 
                    if slack_payload:
                         try:
                            logger.info(f"Posting intermediate message for '{tool_name}' to Slack channel {channel_id}")
                            post_message_to_slack(
                                channel_id=channel_id, 
                                text=slack_payload.get("text", "Formatted tool result."), 
                                blocks=slack_payload.get("blocks"), 
                                thread_ts=thread_ts
                            )
                         except Exception as post_err:
                            logger.error(f"Failed to post intermediate message to Slack: {post_err}")
                    # --- End Intermediate Posting ---                       
                        
                    # Append the ORIGINAL tool result for Bedrock's next action decision
                    if bedrock_content_for_next_step:
                         tool_result_contents.append({
                            "toolResult": {
                                "toolUseId": tool_id,
                                "content": bedrock_content_for_next_step,
                            }
                        })
                    else:
                         logger.error(f"Could not determine bedrock content for tool '{tool_name}' result. Skipping tool result for this request.")

                # Create the user message containing accumulated tool results for this turn
                if not tool_result_contents:
                     logger.warning("No tool results generated for this turn, despite tool_use request.")
                     
                tool_result_message = {
                    "role": "user",
                    "content": tool_result_contents
                }
                messages.append(tool_result_message)

                print(f"--- Tool results: {tool_result_contents} ---")

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

        # Call MCPClient's process_query, passing channel_id and thread_ts
        try:
            final_response = await mcp_client.process_query(user_query, channel_id, thread_ts)
            # Post the final response from Bedrock
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
