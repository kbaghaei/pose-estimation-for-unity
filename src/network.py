# Copyright (c) 2021 Kourosh T. Baghaei
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import socket
import json


class UnityPortal:
    def __init__(self, port_number=65432, message_size=4096):
        self.__port_no = port_number
        self.__ip_addr = '127.0.0.1'

    def send(self, x):
        print('sending data!')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((self.__ip_addr, self.__port_no))
            s.sendall(x)


class ServerSample:
    def __init__(self):
        self.__host = "127.0.0.1"  # Standard loopback interface address (localhost)
        self.__port = 65432  # Port to listen on (non-privileged ports are > 1023)

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self.__host, self.__port))
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    print("Connected by", addr)
                    data = conn.recv(256 * 4 * 16 * 32)
                    if not data:
                        print('no data!')
                        break
                    #    break
                    conn.sendall(data)