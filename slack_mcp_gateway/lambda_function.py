import boto3
import json

lambda_client = boto3.client('lambda')

def lambda_handler(event, context):
    lambda_client.invoke(
        FunctionName='slack-mcp-client',
        InvocationType='Event',
        Payload=json.dumps(event)
    )
    return {
        'statusCode': 200,
        'body': 'Message received'
    }
