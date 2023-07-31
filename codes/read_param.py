"""read parameter file
in this file # is comment symbol
the equal (=) sign is used for geting the value for the key
All the info must be in capital
"""

import sys
import typing
import json
import my_tools
import logger
from colors_text import TextColor as bcolors


class ReadParam:
    """read param file here"""
    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    fname: str = 'update_param'
    essential_key: list[str] = [
        "ANGLE", "RADIUS", "READ", "FILE", "INTERFACE", "INTERFACE_WIDTH",
        "NUMSAMPLE", "ION_DISTANCE", "ION_ATTEPTS", "NP_ZLOC", "LINE",
        "BETTER_POS", "NP_ITP", "APTES_NAMES"]

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.param: dict[str, typing.Any] = {}
        self.get_param(log)
        self.__write_msg(log)
        self.info_msg = ''  # Empety the msg

    def get_param(self,
                  log: logger.logging.Logger
                  ) -> None:
        """check the file and read it"""
        my_tools.check_file_exist(self.fname, log)
        self.read_param()
        self.__sanity_check()

    def read_param(self) -> None:
        """read the file"""
        with open(self.fname, 'r', encoding='utf8') as f_r:
            while True:
                line = f_r.readline()
                if not line.strip().startswith("@"):
                    pass
                else:
                    key, vlaue = self.__process_line(line.strip())
                    self.param[key] = vlaue
                if not line:
                    break
        self.info_msg += f'\tThe parameters read from {self.fname}:\n'
        self.info_msg += json.dumps(self.param, indent=4)

    @staticmethod
    def __process_line(line: str  # Line that read from the file
                       ) -> tuple[str, typing.Any]:
        """process line by spliting by ="""
        line = my_tools.drop_string(line, "@")
        data = line.split('=')
        try:
            return data[0], float(data[1])
        except ValueError:
            return data[0], data[1]

    def __sanity_check(self) -> None:
        """check if all the important keys are exist and have value"""
        # Check if all items in the list are keys in the dictionary
        if not all(item in self.param for item in self.essential_key):
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     f'\tNot all the information provided in `{self.fname}`'
                     f'{bcolors.ENDC}')

    def __write_msg(self,
                    log: logger.logging.Logger
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadParam.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    read = ReadParam(log=logger.setup_logger('update.log'))
