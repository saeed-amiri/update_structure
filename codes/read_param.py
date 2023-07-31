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
        "BETTER_POS", "NP_ITP"]

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
        """
        Check the file existence, read its content, and perform a
        sanity check on the parameters.

        Parameters:
            log (Logger): The logger object to log messages.
        """
        # Check if the file exists, log the message if it does not.
        my_tools.check_file_exist(self.fname, log)

        # Read the parameters from the file and store them in
        # self.param dictionary.
        self.read_param()

        # Perform a sanity check on the read parameters to ensure that
        # all important keys exist.
        self.__sanity_check()

    def read_param(self) -> None:
        """
        Read the file and store the key-value pairs in self.param
        dictionary.
        """
        # Open the file in read mode
        with open(self.fname, 'r', encoding='utf8') as f_r:
            # Read the file line by line until the end
            while True:
                line = f_r.readline()

                # If the line does not start with "@", it is not a
                # parameter, so skip it.
                if not line.strip().startswith("@"):
                    pass
                else:
                    # Process the line to extract key-value pair and
                    # store in self.param
                    key, value = self.__process_line(line.strip())
                    self.param[key] = value

                # If end of file is reached, break the loop
                if not line:
                    break

        # Log the message containing the parameters read from the file.
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
        """
        Check if all the important keys exist and have a value.

        Raises:
            SystemExit: If any of the important keys are missing in
            the param dictionary.
        """
        # Find the missing items from essential_key in param dictionary
        missing_items = \
            [item for item in self.essential_key if item not in self.param]

        # Check if there are any missing items
        if missing_items:
            # Join the missing items into a comma-separated string
            missing_items_str = ', '.join(missing_items)

            # Exit the program and display an error message
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     '\tThe following information is missing in '
                     f'`{self.fname}`:\n\t{missing_items_str}'
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
