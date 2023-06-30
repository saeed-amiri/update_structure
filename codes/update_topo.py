"""read and update the `topol.top` file.
The molecule section in the file should be updated. Only the number of
ions, since the nanoparticle data read as one molecule, the number
there does not need to be updated."""

import my_tools
import logger


class ReadTop:
    """read the topol.top file"""
    def __init__(self,
                 ionnr: int,  # Number of ions
                 ionname: str,  # Name of ion
                 fname: str = 'topol.top'  # Name of the input file
                 ) -> None:
        self.ionnr = ionnr
        self.read_topo(fname, ionname)

    def read_topo(self,
                  fname: str,  # Name of the topofile
                  ionname: str,  # Name of ion
                  ) -> None:
        """check and read the topo file"""
        log = logger.setup_logger('update.log')
        my_tools.check_file_exist(fname, log=log)
        updated_f: str = self.__mk_out_name(fname, log)
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
        return f'{ionname:<15}{self.ionnr:>10}\n'

    @staticmethod
    def __mk_out_name(fname: str,  # Name of the input file
                      log: logger.logging.Logger
                      ) -> str:
        """make a output file name"""
        out_name: str = f"{fname.split('.')[0]}_updated.{fname.split('.')[1]}"
        log.info(f'Update: {fname} -> {out_name}')
        return out_name


if __name__ == '__main__':
    ReadTop(ionnr=200, ionname='CLA')
