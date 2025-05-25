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
    
def get_thread_history(channel_id: str, thread_ts: str) -> List[dict]:
    """
    특정 스레드의 모든 메시지를 가져옵니다.
    """
    if not SLACK_BOT_TOKEN:
        logger.warning("SLACK_BOT_TOKEN not set, cannot fetch thread history")
        return []
    
    try:
        client = WebClient(token=SLACK_BOT_TOKEN)
        
        # conversations.replies API 사용
        response = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            inclusive=True  # 루트 메시지도 포함
        )
        
        messages = response['messages']
        logger.info(f"Fetched {len(messages)} messages from thread {thread_ts}")
        return messages
        
    except SlackApiError as e:
        logger.error(f"Error fetching thread history: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching thread history: {e}")
        return []

def format_thread_for_bedrock(messages: List[dict], bot_user_id: Optional[str] = None) -> List[dict]:
    """
    Slack 메시지들을 Bedrock conversation format으로 변환
    """
    conversation = []
    
    for msg in messages:
        # 봇 메시지인지 사용자 메시지인지 구분
        if msg.get('bot_id') or (bot_user_id and msg.get('user') == bot_user_id):
            role = "assistant"
        else:
            role = "user"
        
        # 텍스트 내용 추출 및 정리
        text = clean_slack_message(msg.get('text', ''))
        
        if text.strip():  # 빈 메시지 제외
            conversation.append({
                "role": role,
                "content": [{"text": text}]
            })
    
    return conversation

def clean_slack_message(text: str) -> str:
    """Slack 포맷팅 제거 (mentions, channels 등)"""
    import re
    
    # <@U123456> 형태의 사용자 멘션 제거
    text = re.sub(r'<@[UW][A-Z0-9]+>', '', text)
    
    # <#C123456|channel-name> 형태의 채널 멘션 정리
    text = re.sub(r'<#[C][A-Z0-9]+\|([^>]+)>', r'#\1', text)
    
    # URL 형태 정리 <http://example.com|example.com>
    text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    
    return text.strip()