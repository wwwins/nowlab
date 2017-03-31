# isobard.py
# read rfid -> pkill -> get user info -> exec face detecter
#
# connect to it with:
#       echo "0001864282 edith"|nc -w 0 localhost 5555
#

from socket import *
from time import sleep
import subprocess

def exec_face_detecter(data):
    subprocess.call(['/usr/local/bin/isobar-face-detect.sh', data])

def pkill(pname):
    subprocess.call(['pkill', '-f', pname])

def get_user_info():
    return "myuserinfo"

def main():
    print("server starting")
    sock = socket()
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR,1)
    sock.bind(('',5555))
    sock.listen(5)
    while True:
        client, addr = sock.accept()
        print('Connection', addr)
        data = client.recv(100)
        if len(data) > 0:
            print('recv:',data)
            pkill("isobar-face-detect.py")
            sleep(0.2)
            exec_face_detecter(data)

if __name__ == '__main__':
    main()
