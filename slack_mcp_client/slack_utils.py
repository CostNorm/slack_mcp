import logging
import json
import os
from typing import Optional, List, Dict, Any

from slack_sdk import WebClient # Uncommented
from slack_sdk.errors import SlackApiError # Uncommented
from slack_sdk.signature import SignatureVerifier # Uncomment when implementing

logger = logging.getLogger()

# --- Environment Variables (Needed by Slack Utils) ---
SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN') # For calling Slack APIs
SLACK_SIGNING_SECRET = os.environ.get('SLACK_SIGNING_SECRET') # For verifying requests

def verify_slack_request(event):
    """
    Verifies the request signature from Slack.
    Replace this placeholder with actual verification using SLACK_SIGNING_SECRET.
    See: https://api.slack.com/authentication/verifying-requests-from-slack
    """
    # Implementation using slack_sdk.signature.SignatureVerifier is recommended
    # verifier = SignatureVerifier(SLACK_SIGNING_SECRET)
    # headers = event.get('headers', {})
    # signature = headers.get('x-slack-signature') # Note: header names are lowercased by API Gateway
    # timestamp = headers.get('x-slack-request-timestamp')
    # request_body = event.get('body', '')
    #
    # if not signature or not timestamp:
    #     logger.warning("Missing Slack signature headers.")
    #     return False
    #
    # if not verifier.is_valid_request(request_body, headers):
    #     logger.warning("Invalid Slack signature.")
    #     return False
    # logger.info("Slack request signature verified.")

    logger.warning("Slack request verification SKIPPED - implement actual logic!")
    return True # Assume verified for now - Replace in production!

def post_message_to_slack(channel_id: str, text: str, thread_ts: Optional[str] = None, blocks: Optional[List[Dict[str, Any]]] = None):
    """
    Posts a message to a Slack channel, optionally replying in a thread.
    Can send plain text or messages with Block Kit layouts.
    Requires SLACK_BOT_TOKEN with chat:write scope.
    
    Args:
        channel_id: The ID of the channel to post to.
        text: The plain text summary of the message (used for notifications).
        thread_ts: The timestamp of the parent message to reply in a thread (optional).
        blocks: A list of Block Kit layout blocks (optional).
    """
    logger.info(f"Attempting to post to Slack channel {channel_id} (Thread: {thread_ts})")
    if not channel_id or (not text and not blocks): # Need at least text or blocks
        logger.error("Missing channel_id or message content (text or blocks) for posting.")
        return
    if not text and blocks:
         logger.warning("Posting message with blocks but no fallback text. Notifications may be unclear.")
         text = "Structured message content." # Provide a minimal default fallback

    # --- Implementation using slack_sdk --- 
    try:
        if not SLACK_BOT_TOKEN:
             raise ValueError("SLACK_BOT_TOKEN environment variable not set.")
        client = WebClient(token=SLACK_BOT_TOKEN)
        
        # Prepare arguments for API call
        api_args = {
            "channel": channel_id,
            "text": text, # Always include text for fallback/notifications
            "thread_ts": thread_ts
        }
        # Add blocks if provided and valid
        if blocks and isinstance(blocks, list) and len(blocks) > 0:
            api_args["blocks"] = blocks
            logger.debug(f"Posting message with {len(blocks)} blocks.")
        else:
             logger.debug("Posting message with text only.")
            
        response = client.chat_postMessage(**api_args)
        
        logger.info(f"Message posted to Slack: {response['ts']}")
    except SlackApiError as e:
        logger.error(f"Error posting message to Slack: {e.response['error']}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while posting to Slack: {e}", exc_info=True)

    # logger.warning("Slack message posting SKIPPED - implement actual logic!") # Removed placeholder warning
    # pass # Removed placeholder pass

def parse_slack_event_body(event):
    """
    Parses the JSON body from the Lambda event triggered by API Gateway.
    """
    try:
        body = json.loads(event.get('body', '{}'))
        return body
    except json.JSONDecodeError:
        logger.error("Failed to parse request body as JSON.")
        return None 