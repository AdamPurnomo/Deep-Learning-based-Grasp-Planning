import socket
import time 

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5)

while True:
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been establised.")
    action_done_flag = int(clientsocket.recv(1024).decode("utf-8"))
    clientsocket.close()
    print("Action Done")

    if(action_done_flag==1):
        print("Inference . . .")
        time.sleep(10)
    
    clientsocket, addres = s.accept()
    clientsocket.send(bytes("0", "utf-8"))
    print("Action Start")
    clientsocket.close()

    
