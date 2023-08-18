"""
Find where the script is running and set the number of available cores
that can be used.
"""

import socket
import multiprocessing
import logger
from colors_text import TextColor as bcolors


class ConfigCpuNr:
    """
    Find the number of core
    """

    info_msg: str = 'message from ConfigCpuNr:\n' # Meesage from methods to log

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.hostname: str = self.get_hostname()
        self.core_nr: int = self.get_core_nr()
        self.write_log_msg(log)

    def get_hostname(self) -> str:
        """find the name of the host"""
        hostname = socket.gethostname()
        self.info_msg += f'\t\tHostname is `{hostname}`\n'
        return hostname

    def get_core_nr(self) -> int:
        """return numbers of cores"""
        core_nr: int = multiprocessing.cpu_count()
        self.info_msg += f'\t\tNumber of cores is: `{core_nr}`\n'
        return core_nr

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')

if __name__ == '__main__':
    ConfigCpuNr(log=logger.setup_logger(log_name='ConfigeCpu.log'))
