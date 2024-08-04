import os
import signal
import subprocess

import typer

app = typer.Typer()

SOCKET_PORT_START = 9000
CONFIG_PORT_START = 9900

processes = []

def create_torrc_file(index: int, socks_port: int, control_port: int, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    config_path = f"/etc/tor/torrc.{index}"
    with open(config_path, "w") as file:
        file.write(f"SocksPort 0.0.0.0:{socks_port}\n")
        file.write(f"ControlPort {control_port}\n")
        file.write(f"DataDirectory {data_dir}\n")

def create_configs(num_tors: int) -> None:
    configs = []
    for i in range(num_tors):
        socks_port = SOCKET_PORT_START + i
        control_port = CONFIG_PORT_START + i
        data_dir = f"/var/lib/tor{i}"
        configs.append((i, socks_port, control_port, data_dir))

    for index, socks_port, control_port, data_dir in configs:
        create_torrc_file(index, socks_port, control_port, data_dir)

def signal_handler(sig, frame):
    print("Terminating all processes...")
    for process in processes:
        process.terminate()
    for process in processes:
        process.wait()
    print("All processes terminated.")
    exit(0)

def run_tor_instances(num_tors: int = typer.Option(5)) -> None:
    create_configs(num_tors)
    signal.signal(signal.SIGINT, signal_handler)
    for i in range(num_tors):
        config_path = f"/etc/tor/torrc.{i}"
        process = subprocess.Popen(["tor", "-f", config_path])
        processes.append(process)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    typer.run(run_tor_instances)
