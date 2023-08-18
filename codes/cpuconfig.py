"""
Find where the script is running and set the number of available cores
that can be used.
"""

import socket
import multiprocessing

class ConfigCpuNr:
    """
    Find the number of core
    """
    def __init__(self) -> None:
        self.hostname: str = self.get_hostname()
        self.core_nr: int = self.get_core_nr()

    def get_hostname(self) -> str:
        """find the name of the host"""
        hostname = socket.gethostname()
        print(hostname)

    def get_core_nr(self) -> int:
        """return numbers of cores"""
        core_nr: int = multiprocessing.cpu_count()
        print(core_nr)


if __name__ == '__main__':
    ConfigCpuNr()
