from os import system

COMMAND = "uvicorn main:app --host 127.0.0.1 --port 80"

system(COMMAND)
