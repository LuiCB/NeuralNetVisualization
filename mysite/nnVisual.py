import os
import subprocess

print(os.getcwd())

def runServer():
	subprocess.run(['python3', 'manage.py', 'runserver', '8080'])
	print("tset")

if __name__ == "__main__":
	runServer()