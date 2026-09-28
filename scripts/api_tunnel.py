"""企业匹配 API 的 SSH 隧道（to GPU server）：本机端口 -> 服务器 8022。

用途：GPU 服务器上的企业匹配 API 绑 0.0.0.0 但外部网络 ACL 未放行 8022；
SSH(22) 可达时，用本脚本把本机 127.0.0.1:<本地端口> 转发到服务器 127.0.0.1:8022。

用法：
    set TUNNEL_HOST=192.168.1.52
    set TUNNEL_USER=ubuntu
    set TUNNEL_PASSWORD=***
    python scripts/api_tunnel.py --local-port 8021
    # 然后访问 http://127.0.0.1:8021/（调试台）或 POST /api/match

长期方案：在网关/安全组放行 8022/tcp 后即可直连 192.168.1.52:8022。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import paramiko


def _forward(local_port: int, remote_port: int, host: str, user: str, password: str) -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=15)
    transport = ssh.get_transport()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", local_port))
    server.listen(64)
    print(f"[tunnel] 127.0.0.1:{local_port} -> {host}:127.0.0.1:{remote_port} (SSH)")
    print("[tunnel] Ctrl+C 退出")

    def handle(conn):
        try:
            chan = transport.open_channel("direct-tcpip",
                                          ("127.0.0.1", remote_port), conn.getpeername())

            def pump(src, dst):
                try:
                    while True:
                        data = src.recv(65536)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass
                finally:
                    try:
                        dst.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass

            threading.Thread(target=pump, args=(conn, chan), daemon=True).start()
            threading.Thread(target=pump, args=(chan, conn), daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            conn.close()

    while True:
        try:
            conn, _ = server.accept()
        except OSError:
            break
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description="SSH 隧道到 GPU 服务器企业匹配 API (8022)")
    ap.add_argument("--local-port", type=int, default=8021)
    ap.add_argument("--remote-port", type=int, default=8022)
    ap.add_argument("--host", default=os.environ.get("TUNNEL_HOST", "192.168.1.52"))
    ap.add_argument("--user", default=os.environ.get("TUNNEL_USER", "ubuntu"))
    ap.add_argument("--password", default=os.environ.get("TUNNEL_PASSWORD", ""))
    args = ap.parse_args()
    if not args.password:
        import getpass
        args.password = getpass.getpass("TUNNEL_PASSWORD: ")
    _forward(args.local_port, args.remote_port, args.host, args.user, args.password)


if __name__ == "__main__":
    main()