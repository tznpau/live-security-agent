"""
watcher.py - standalone test of SSH tail -> queue -> timed/cap drain.
Run this alone first, no Streamlit, to validate timing against the real Pi.
"""
import paramiko
import queue
import threading
import time
import getpass

PI_HOST = "pifive"
PI_USER = "piadmin"
PI_PASSWORD = getpass.getpass(f"Password for {PI_USER}@{PI_HOST}: ")
LOG_PATH = "/home/piadmin/zeek-logs/live/conn.log"

DURATION_SECONDS = 25
FLOW_CAP = 30


def tail_worker(out_queue: queue.Queue, stop_event: threading.Event):
    """Background thread: opens SSH, starts `tail -F` on the Pi, pushes
    every line into out_queue until stop_event is set."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=PI_HOST,
        username=PI_USER,
        password=PI_PASSWORD,
        look_for_keys=False,
        allow_agent=False,
    )

    # get_pty=False keeps stdout clean of terminal escape sequences
    _, stdout, _ = client.exec_command(f"tail -F {LOG_PATH}", get_pty=False)

    channel = stdout.channel
    channel.settimeout(1.0)  # avoids blocking forever on recv()

    buffer = ""
    while not stop_event.is_set():
        if channel.recv_ready():
            buffer += channel.recv(4096).decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.startswith("#"):
                    continue  # Zeek metadata — header lines or #close, not a flow record
                out_queue.put(line)
        else:
            time.sleep(0.1)

    channel.close()
    client.close()


def run_capture():
    """Main thread: drains the queue for DURATION_SECONDS or until
    FLOW_CAP lines arrive, whichever happens first."""
    q = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=tail_worker, args=(q, stop_event), daemon=True)
    thread.start()

    collected = []
    start = time.time()

    while time.time() - start < DURATION_SECONDS and len(collected) < FLOW_CAP:
        try:
            line = q.get(timeout=0.5)
            collected.append(line)
            print(f"[{len(collected):02d}] {line}")
        except queue.Empty:
            continue

    stop_event.set()
    thread.join(timeout=2)
    print(f"\nStopped after {time.time() - start:.1f}s, {len(collected)} lines captured.")
    return collected


if __name__ == "__main__":
    run_capture()