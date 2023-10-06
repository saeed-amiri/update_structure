"""
Read Parameter File

This script reads a parameter file and extracts key-value pairs from
it. The parameter file is a text file with lines of
the form 'key = value'. Lines starting with '#' are considered comments
and ignored. All the keys in the parameter file must be in capital
letters.

The script performs the following steps:
    1. Check the existence of the parameter file.
    2. Read the parameter file and store the key-value pairs in a
    dictionary.
    3. Perform a sanity check to ensure that all essential keys are
    present in the parameter dictionary.

Usage:
    The script can be executed directly, and it takes an optional
    argument 'fname', which specifies the name of the parameter file.
    If 'fname' is not provided, it defaults to 'update_param'.
    The script will log information and errors to the 'update.log' file.

Classes:
    ReadParam:
        The main class that reads the parameter file, extracts
        key-value pairs, and performs the sanity check on essential
        keys.

        Attributes:
            fname (str): The name of the parameter file.
            param (dict): A dictionary to store the key-value pairs
            from the parameter file.
            log (Logger): The logger object for logging messages.
            info_msg (str): A message string for logging and writing
            information.
            essential_keys (list[str]): A list of essential keys that
            must exist in the parameter dictionary.

        Methods:
            __init__(self, fname='update_param', log=None):
                Initialize the ReadParam object.

            load_param_from_file(self):
                Check the existence of the parameter file, read its
                content, and load parameters from the file.

            read_param_file(self):
                Read the parameter file and store the key-value pairs
                in self.param dictionary.

            process_line(self, line: str) -> tuple[str, typing.Any]:
                Process a line from the parameter file and extract
                the key-value pair.

            check_essential_keys_exist(self):
                Check if all the essential keys exist in the parameter
                dictionary.

            write_log_message(self):
                Write and log messages after processing the parameter
                file.

    Dependencies:
    The script relies on several external modules, including:
     'typing', 'json', 'my_tools', 'logger', and 'colors_text',

    Example:
    The script can be executed directly to read the 'update_param'
    file and log the extracted key-value pairs:
        if __name__ == '__main__':
            read = ReadParam(log=logger.setup_logger('update.log'))
"""

import sys
import json
import typing
import logger
import my_tools
from colors_text import TextColor as bcolors


class ReadParam:
    """read parameter file
    in this file # is comment symbol
    the equal (=) sign is used for geting the value for the key
    All the info must be in capital
    """
    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    fname: str = 'update_param'
    float_keys: list[str] = ["ANGLE", "RADIUS", "INTERFACE",
                             "INTERFACE_WIDTH", "ION_DISTANCE", "NP_ZLOC"]
    integer_keys: list[str] = ["NUMSAMPLE", "ION_ATTEMPTS", "BETTER_POS"]
    boolen_keys: list[str] = ["READ", "DEBUG"]
    files_keys: list[str] = ["TOPOFILE"]
    itp_keys: list[str] = ["NP_ITP"]
    str_keys: list[str] = ["FILE", "LINE"]
    list_keys: list = [float_keys, integer_keys, boolen_keys, files_keys,
                       str_keys, itp_keys]
    optional_keys: list[str] = ["NUMAPTES"]

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.param: dict[str, typing.Any] = {}
        self.load_param_from_file(log)
        self.get_names_of_parts()
        self.check_inputs(log)
        self.__write_msg(log)
        self.info_msg = ''  # Empety the msg

    def load_param_from_file(self,
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

        self.check_optinal_keys()
        # Perform a sanity check on the read parameters to ensure that
        # all important keys exist.
        essential_keys = \
            [item for sublist in self.list_keys for item in sublist]
        self.check_essential_keys_exist(essential_keys)

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

    def get_names_of_parts(self) -> None:
        """
        Find the names of the APTES and CORE atoms and also ITP files.

        This method extracts the names of the APTES and CORE atoms, as
        well as the names of the ITP (GROMACS topology) files from the
        `NP_ITP` parameter in the input file.
        The extracted names are then stored in the `aptes`, `cores`,
        and `itp_files` fields of the `param` dictionary.

        Note: The `NP_ITP` parameter in the input file is expected to
        be a semicolon-separated list of values in the format:
        '[ITP_FILE_NAME], [APTES_NAME], [CORE_NAME]; [ITP_FILE_NAME],
        [APTES_NAME], [CORE_NAME]; ...'

        Raises:
            ValueError: If the `NP_ITP` parameter is not in the correct
                        format.
        """
        files_list: list[str] = []  # To save all the itp files names
        aptes_list: list[str] = []  # To save all the aptes names
        cores_list: list[str] = []  # To save all the cores names
        files = self.param.get('NP_ITP', '').split(';')
        for itp in files:
            itp = itp.strip()
            itps = my_tools.drop_string(itp, '[', ']')
            names = itps.split(',')
            files_list.append(names[0].strip() + '.itp')
            aptes_list.append(names[1].strip())
            cores_list.append(names[2].strip())
        self.param['itp_files'] = files_list
        self.param['aptes'] = aptes_list
        self.param['cores'] = cores_list
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

    def check_essential_keys_exist(self,
                                   essential_keys: list[str]
                                   ) -> None:
        """
        Check if all the important keys exist and have a value.

        Raises:
            SystemExit: If any of the important keys are missing in
            the param dictionary.
        """
        # Find the missing items from essential_keys in param dictionary
        missing_items = \
            [item for item in essential_keys if item not in self.param]

        # Check if there are any missing items
        if missing_items:
            # Join the missing items into a comma-separated string
            missing_items_str = ', '.join(missing_items)

            # Exit the program and display an error message
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     '\tThe following information is missing in '
                     f'`{self.fname}`:\n\t{missing_items_str}'
                     f'{bcolors.ENDC}')

    def check_inputs(self,
                     log: logger.logging.Logger
                     ) -> None:
        """
        Check the input of parameter file. Check numbers, files,...
        """
        self.check_files([self.param[item] for item in self.files_keys], log)
        self.check_files(self.param['itp_files'], log)

    def check_optinal_keys(self) -> None:
        """check if the optinal keys have values, otherwise give them
        -1
        """
        for item in self.optional_keys:
            if item not in self.param:
                self.param[item] = -1
                self.info_msg += f'\tThe value for {item} is set to `-1`\n'

    @staticmethod
    def check_files(file_list: list[str],
                    log: logger.logging.Logger
                    ) -> None:
        """check all the files if their exist"""
        for ifile in file_list:
            my_tools.check_file_exist(ifile, log, logging=False)

    def __write_msg(self,
                    log: logger.logging.Logger
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadParam.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    read = ReadParam(log=logger.setup_logger('read_parm.log'))
