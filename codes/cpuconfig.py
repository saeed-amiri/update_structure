"""
Find where the script is running and set the number of available cores
that can be used.
"""

import socket

class ConfigCpuNr:
    """
    Find the number of core
    """
    def __init__(self) -> None:
        self.get_hostname()

    def get_hostname(self) -> str:
        """find the name of the host"""
        hostname = socket.gethostname()
        print(hostname)


if __name__ == '__main__':
    ConfigCpuNr()
