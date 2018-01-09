import os
import subprocess
import sys

print(os.getcwd())

def runServer(port):
	subprocess.run(['python3', 'manage.py', 'runserver', port])

if __name__ == "__main__":
	if (len(sys.argv) != 2):
		port = '8080'
	else:
		port = sys.argv[1] 
	runServer(port)