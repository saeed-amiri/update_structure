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

    info_msg: str = 'message from ConfigCpuNr:\n'  # Meesage in methods to log
    server_front_host: str = 'glogin'  # Name of the goettingen front of HLRN
    local_host: str = 'hmigws03'  # Name of the host in the office
    # Name of the goettingen of HLRN
    server_host_list: list[str] = ['gcn', 'gfn', 'gsn', 'bcn', 'bfn', 'bsn']

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.hostname: str = self.get_hostname()
        self.core_nr: int = self.set_core_numbers()
        self.write_log_msg(log)

    def set_core_numbers(self) -> int:
        """set the nmbers of the cores based on the hostname"""
        aval_core_nr: int = self.get_core_nr()
        if self.hostname == self.local_host:
            # In local machine only using half of the cores
            core_nr = aval_core_nr // 2
        elif self.hostname == self.server_front_host:
            # On frontend use only 4 since it is for all
            core_nr = 4
        elif self.hostname[:3] in self.server_host_list:
            # On the backends use all the physical cores
            core_nr = aval_core_nr // 2
        else:
            core_nr = aval_core_nr
        self.info_msg += (f'\t\tNumber of cores for this computation is'
                          f' set to: `{core_nr}`\n')
        return core_nr

    def get_hostname(self) -> str:
        """find the name of the host"""
        hostname = socket.gethostname()
        self.info_msg += f'\t\tHostname is `{hostname}`\n'
        return hostname

    def get_core_nr(self) -> int:
        """return numbers of cores"""
        aval_core_nr: int = multiprocessing.cpu_count()
        self.info_msg += \
            f'\t\tNumber of available cores of the host is: `{aval_core_nr}`\n'
        return aval_core_nr

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    ConfigCpuNr(log=logger.setup_logger(log_name='ConfigeCpu.log'))
