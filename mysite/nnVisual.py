import os
import subprocess
import sys

print(os.getcwd())

def runServer(port):
	subprocess.run(['python3', 'manage.py', 'runserver', port])

if __name__ == "__main__":
	port = sys.argv[1]
	print(port) 
	runServer(port)