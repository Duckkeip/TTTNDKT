import os
from twilio.rest import Client
from dotenv import load_dotenv

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

print(account_sid)

client = Client(account_sid, auth_token)
token = client.tokens.create()

print(token.ice_servers)