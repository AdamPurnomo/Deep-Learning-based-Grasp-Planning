import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))
action_done_flag = int(s.recv(1024).decode("utf-8"))
if(action_done_flag == 0):
    print("Action Start")