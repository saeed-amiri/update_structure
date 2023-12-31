"""write itp files for updated sections"""

import typing
import pandas as pd
import logger
from colors_text import TextColor as bcolors
if typing.TYPE_CHECKING:
    from update_itp import UpdateItp, WrapperUpdateItp


class WriteItp:
    """write itp file
    There is no uniqe structure one should follow
    The columns will be seperated by single space"""

    info_msg = 'Message from WriteItp:\n'

    def __init__(self,
                 itp: 'UpdateItp',  # Dataframes of updated sections
                 fname: str,  # Name of the updated itp file
                 log: logger.logging.Logger
                 ) -> None:
        """call functions"""
        self.fname: str = fname  # Name of the itp file making in the class
        self.write_itp(itp)
        self.__write_msg(log)
        self.info_msg = ''  # clean the msg

    def write_itp(self,
                  itp: 'UpdateItp'  # Dataframes of updated sections
                  ) -> None:
        """write itp file for all the residues, and return the name of itp"""
        with open(self.fname, 'w', encoding="utf8") as f_w:
            f_w.write('; input pdb SMILES:\n')
            f_w.write('; Updated itp file for nanoparticle:\n')
            f_w.write('\n')
            self.write_molecule(f_w, itp.molecules)
            self.write_atoms(f_w, itp.atoms_updated)
            self.write_bonds(f_w, itp.bonds_updated)
            self.write_angles(f_w, itp.angles_updated)
            self.write_dihedrals(f_w, itp.dihedrals_updated)
            self.info_msg += f'\tThe updated NP itp file is: `{self.fname}`\n'

    @staticmethod
    def write_molecule(f_w: typing.Any,  # The out put file
                       molecule: pd.DataFrame  # Molecule section
                       ) -> None:
        """write section of the itp file"""
        header: list[str] = molecule.columns.to_list()
        f_w.write('[ moleculetype ]\n')
        f_w.write(f'; {"  ".join(header)}\n')
        molecule.to_csv(f_w,
                        header=None,
                        sep='\t',
                        index=False)
        f_w.write('\n')

    @staticmethod
    def write_atoms(f_w: typing.Any,  # The out put file
                    atoms: pd.DataFrame  # Atoms information
                    ) -> None:
        """write atom section of the itp file"""
        header: list[str] = list(atoms.columns)  # Header of atoms
        f_w.write('[ atoms ]\n')
        f_w.write(f'; {"  ".join(header)}\n')
        atoms[';'] = [';' for _ in atoms.index]
        for row in atoms.iterrows():
            line: list[str]  # line with length of 85 spaces to fix output
            line = [' '*85]
            line[0:7] = f'{row[1]["atomnr"]:>7d}'
            line[7:11] = f'{" "*2}'
            line[11:19] = f'{row[1]["atomtype"]:>7s}'
            line[19:21] = f'{" "*2}'
            line[21:26] = f'{row[1]["resnr"]:5d}'
            line[26:28] = f'{" "*2}'
            line[28:35] = f'{row[1]["resname"]:>7s}'
            line[35:37] = f'{" "*2}'
            line[37:45] = f'{row[1]["atomname"]:>8s}'
            line[45:47] = f'{" "*2}'
            line[47:56] = f'{row[1]["chargegrp"]:>9s}'
            line[56:58] = f'{" "*2}'
            line[58:64] = f'{row[1]["charge"]:>6.3f}'
            line[64:66] = f'{" "*2}'
            line[66:73] = f'{row[1]["mass"]:>6s}'
            line[73:74] = f'{" "*1}'
            line[75:77] = f'{row[1][";"]:>2s}'
            line[77:78] = f'{" "*1}'
            line[78:] = f'{row[1]["element"]:>6s}'
            f_w.write(''.join(line))
            f_w.write('\n')
        f_w.write(f'; Total charge : {atoms["charge"].sum()}\n')
        f_w.write('\n')

    @staticmethod
    def write_bonds(f_w: typing.Any,  # The out put file
                    bonds: pd.DataFrame,  # bonds information
                    ) -> None:
        """write bonds section of the itp file"""
        header: list[str] = list(bonds.columns)
        f_w.write('[ bonds ]\n')
        f_w.write(f'; {" ".join(header)}\n')
        bonds.to_csv(f_w,
                     header=None,
                     sep='\t',
                     index=False,
                     float_format='%.5f')
        f_w.write('\n')

    @staticmethod
    def write_angles(f_w: typing.Any,  # The out put file
                     angles: pd.DataFrame  # Angles inoformation
                     ) -> None:
        """write section of the itp file"""
        header: list[str] = list(angles.columns)
        f_w.write('[ angles ]\n')
        f_w.write(f'; {" ".join(header)}\n')
        angles.to_csv(f_w,
                      header=None,
                      sep='\t',
                      index=False,
                      float_format='%.5f')
        f_w.write('\n')

    @staticmethod
    def write_dihedrals(f_w: typing.Any,  # The out put file
                        dihedrals: pd.DataFrame,  # Dihedrals inoformation
                        ) -> None:
        """write section of the itp file"""
        header: list[str] = list(dihedrals.columns)
        f_w.write('[ dihedrals ]\n')
        f_w.write(f'; {" ".join(header)}\n')
        dihedrals.to_csv(f_w,
                         header=None,
                         sep='\t',
                         index=False,
                         float_format='%.5f')
        f_w.write('\n')

    def __write_msg(self,
                    log: logger.logging.Logger
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{WriteItp.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class WrapperWriteItp:
    """
    A wrapper class to cover all the itp files
    """
    def __init__(self,
                 itps: 'WrapperUpdateItp',
                 log: logger.logging.Logger
                 ) -> None:
        self.write_itp_files(itps, log)

    def write_itp_files(self,
                        itps: 'WrapperUpdateItp',
                        log: logger.logging.Logger
                        ) -> None:
        """loop over all the itp files"""
        for nano_p, itp_item in itps.updated_itp.items():
            WriteItp(itp_item, self.mk_out_name(nano_p), log)

    @staticmethod
    def mk_out_name(nano_p: str) -> str:
        """make an output file name"""
        ouput_filename = nano_p.split('.')[0] + '_updated.itp'
        return ouput_filename


if __name__ == '__main__':
    pass
