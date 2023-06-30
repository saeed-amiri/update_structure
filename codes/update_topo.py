"""read and update the `topol.top` file.
The molecule section in the file should be updated. Only the number of
ions, since the nanoparticle data read as one molecule, the number
there does not need to be updated."""

import my_tools
import logger
from colors_text import TextColor as bcolors


class ReadTop:
    """read the topol.top file"""

    info_msg: str = 'message:\n' # Meesage from methods to log at the end

    def __init__(self,
                 ionnr: int,  # Number of ions
                 ionname: str,  # Name of ion
                 log: logger.logging.Logger,  # To log infos
                 fname: str = 'topol.top'  # Name of the input file
                 ) -> None:
        self.ionnr = ionnr
        self.read_topo(fname, ionname, log)
        self.write_log_msg(log)

    def read_topo(self,
                  fname: str,  # Name of the topofile
                  ionname: str,  # Name of ion
                  log: logger.logging.Logger  # To log infos
                  ) -> None:
        """check and read the topo file"""
        my_tools.check_file_exist(fname, log=log)
        updated_f: str = self.__mk_out_name(fname)
        with open(fname, 'r', encoding='utf8') as f_r, \
             open(updated_f, 'w', encoding='utf8') as f_w:
            while True:
                line = f_r.readline()
                if line.strip().startswith(ionname):
                    line = self.__process_line(ionname)
                f_w.write(line)
                if not line:
                    break

    def __process_line(self,
                       ionname: str,  # Name of ion
                       ) -> str:
        """update the number of ions in the file"""
        new_line: str = f'{ionname:<15}{self.ionnr:>10}\n'
        ReadTop.info_msg += f'\tUpdated ion line is: {new_line}'
        return new_line

    def __mk_out_name(self,
                      fname: str,  # Name of the input file
                      ) -> str:
        """make a output file name"""
        out_name: str = f"{fname.split('.')[0]}_updated.{fname.split('.')[1]}"
        ReadTop.info_msg += f'\tUpdate: {fname} -> {out_name}\n'
        return out_name

    @classmethod
    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    
    ReadTop(ionnr=200, ionname='CLA',
            log = logger.setup_logger('update.log'))
