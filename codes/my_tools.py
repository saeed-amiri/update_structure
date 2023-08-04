"""tools used in multiple scripts"""
import os
import re
import sys
import typing
import logger
from colors_text import TextColor as bcolors


def check_file_exist(fname: str,  # Name of the file to check
                     log: logger.logging.Logger,  # log the error
                     logging: bool = True
                     ) -> None:
    """check if the file exist, other wise exit"""
    if not os.path.exists(fname):
        log.error(f'Error! `{fname}` dose not exist.')
        sys.exit(f'{bcolors.FAIL}{__name__}: '
                 f'(Error! `{fname}` dose not '
                 f'exist \n{bcolors.ENDC}')
    else:
        if logging:
            log.info(f'reading: `{fname}`')


def check_file_reanme(fname: str,  # Name of the file to check
                      log: logger.logging.Logger
                      ) -> str:
    """checking if the file fname is exist and if, rename the old one"""
    # Check if the file already exists
    if os.path.isfile(fname):
        # Generate a new file name by appending a counter
        counter = 1
        while os.path.isfile(f"{fname}_{counter}"):
            counter += 1
        new_fname = f"{fname}_{counter}"

        # Rename the existing file
        os.rename(fname, new_fname)
        print(f'{bcolors.CAUTION}{__name__}:\n\tRenaming an old `{fname}` '
              f' file to: `{new_fname}`{bcolors.ENDC}')
        log.info(f'renmaing an old `{fname}` to `{new_fname}`')
    return fname


def drop_string(input_string: str,
                *strings_to_drop: str
                ) -> str:
    """
    Remove all occurrences of a specified substring from an input
    string.

    This function performs a substring replacement in the given
    input_string
    by removing all instances of the specified string_to_drop.
    The resulting modified string is returned.

    Parameters:
        input_string (str): The input string from which the substring
                            will be removed.
        string_to_drop (str): The substring that needs to be removed
                              from the input_string.

    Returns:
        str: A new string with all occurrences of string_to_drop
             removed.
    """
    output_string = input_string
    for string_to_drop in strings_to_drop:
        output_string = output_string.replace(string_to_drop, "")

    # Return the resulting modified string.
    return output_string


def extract_string(input_string: str) -> list[typing.Any]:
    """
    Extract substrings enclosed within double quotes from the input
    string.

    Parameters:
        input_string (str): The string from which to extract substrings.

    Returns:
        List[typing.Any]: A list containing the substrings enclosed
        within double quotes.

    Example:
        >>> result = extract_string('This is a "sample" string with
        "multiple" occurrences of "double-quoted" substrings.')
        >>> print(result)
        ['sample', 'multiple', 'double-quoted']

    Note:
        - The function uses a non-greedy match to extract substrings
        within the shortest possible double quotes pair.
        - The function may return an empty list if no substrings
        enclosed within double quotes are found in the input string.
        - The function is case-sensitive and only considers double
        quotes (") to enclose substrings.
        - It does not handle escaped double quotes within substrings.
    """
    # Regular expression pattern to matchsubstrings within double quotes
    pattern = r'"(.*?)"'
    # Find all occurrences of the pattern
    matches = re.findall(pattern, input_string)
    return matches
