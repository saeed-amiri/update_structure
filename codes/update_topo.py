"""read and update the `topol.top` file.
The molecule section in the file should be updated. Only the number of
ions, since the nanoparticle data read as one molecule, the number
there does not need to be updated."""

import typing
import my_tools
import logger
from colors_text import TextColor as bcolors


class ReadTop:
    """read the topol.top file"""

    info_msg: str = 'message:\n' # Meesage from methods to log at the end

    def __init__(self,
                 nr_atoms_residues: dict[str, int],
                 param: dict[str, typing.Any],
                 log: logger.logging.Logger,
                 ) -> None:
        
        self.read_topo(nr_atoms_residues ,param, log)
        self.write_log_msg(log)

    def read_topo(self,
                  nr_atoms_residues: dict[str, int],
                  param: dict[str, typing.Any],
                  log: logger.logging.Logger,
                  ) -> None:
        """check and read the topo file"""
        fname: str = param['TOPOFILE']
        my_tools.check_file_exist(fname, log=log)
        updated_f: str = self.__mk_out_name(fname)
        with open(fname, 'r', encoding='utf8') as f_r, \
             open(updated_f, 'w', encoding='utf8') as f_w:
            while True:
                line = f_r.readline()
                if not line.strip().startswith('[ molecules ]'):
                    f_w.write(line)
                else:
                    break
                if not line:
                    break
            line = self.write_residues(nr_atoms_residues, f_w)

    def write_residues(self,
                       nr_atoms_residues: dict[str, int],  # Number of residues
                       f_w: typing.IO
                       ) -> None:
        """update the number of ions in the file"""
        f_w.write('[ molecules ]\n')
        f_w.write('; Compound			#mols\n')
        for res, numbers in nr_atoms_residues.items():
            if 'itp' not in res:
                nr: int = numbers['nr_residues']
            else:
                res = res.split('.')[0]
                nr = 1
            new_line: str = f'{res:<15}{nr:>10}\n'
            f_w.write(new_line)
        self.info_msg += f'\tUpdated ion line is: {new_line}'

    def __mk_out_name(self,
                      fname: str,  # Name of the input file
                      ) -> str:
        """make a output file name"""
        out_name: str = f"{fname.split('.')[0]}_updated.{fname.split('.')[1]}"
        self.info_msg += f'\tUpdate: {fname} -> {out_name}\n'
        self.info_msg += \
            '\n\t\tUPDATE THE NAME OF ITP FILES! I DONT DO IT!\n\n'
        return out_name

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    ReadTop(nr_atoms_residues={'SOL': 100, 'ODN': 10},
            param={'TOPOFILE': 'topol.top'},
            log = logger.setup_logger('update_topo.log'))
