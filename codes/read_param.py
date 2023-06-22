"""read parameter file
in this file # is comment symbol
the equal (=) sign is used for geting the value for the key
All the info must be in capital
This file must include:
    ############### CONSTANT VALUES ##############
    # The contact angle of the nanoparticle (NP)
    ANGLE=90
    # Estimate radius of the NP before functionalization with APTES
    RADIUS=25
    ############### INTERACTIVE VALUES ###########
    # intereface location -> caculated by analysing script
    INTERFACE=12
    # interface thickness
    INTERFACE_WIDTH=10
"""

import sys
import typing
import my_tools
import logger
from colors_text import TextColor as bcolors


class ReadParam:
    """read param file here"""
    fname: str = 'update_param'
    essential_key: list[str] = [
        "ANGLE", "RADIUS", "INTERFACE", "INTERFACE_WIDTH", "NUMSAMPLE",
        "ION_DISTANCE", "ION_ATTEPTS"]

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.param: dict[str, typing.Any] = {}
        self.get_param(log)

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

    @staticmethod
    def __process_line(line: str  # Line that read from the file
                       ) -> tuple[str, float]:
        """process line by spliting by ="""
        line = my_tools.drop_string(line, "@")
        data = line.split('=')
        return data[0], float(data[1])

    def __sanity_check(self) -> None:
        """check if all the important keys are exist and have value"""
        # Check if all items in the list are keys in the dictionary
        if not all(item in self.param for item in self.essential_key):
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     f'\tNot all the information provided in `{self.fname}`'
                     f'{bcolors.ENDC}')
        # Check if each key has a float value
        for key, _ in self.param.items():
            if not isinstance(self.param[key], float):
                sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                         f'\tIncorrect essentail value in `{self.fname}`'
                         f'{bcolors.ENDC}')


if __name__ == '__main__':
    read = ReadParam(log=logger.setup_logger('read_param.log'))
