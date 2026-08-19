import configparser
import os
from dotenv import load_dotenv

load_dotenv()

config = configparser.ConfigParser()
config.read("/vault/secrets/creds")

SECRETS_PATH = os.environ.get("SECRETS_PATH", "/vault/secrets/creds")
secrets = configparser.ConfigParser()
ans = secrets.read(SECRETS_PATH)

print(ans)

print("=="*10)

print(secrets)
