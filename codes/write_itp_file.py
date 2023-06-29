"""write itp files for updated sections"""

import typing
import pandas as pd
import update_itp as upitp
from colors_text import TextColor as bcolors


class WriteItp:
    """write itp file
    There is no uniqe structure one should follow
    The columns will be seperated by single space"""
    def __init__(self,
                 itp: upitp.UpdateItp,  # Dataframes of updated sections
                 fname: str  # Name of the updated itp file
                 ) -> None:
        """call functions"""
        self.fname: str = fname  # Name of the itp file making in the class
        self.write_itp(itp)

    def write_itp(self,
                  itp: upitp.UpdateItp  # Dataframes of updated sections
                  ) -> None:
        """write itp file for all the residues, and return the name of itp"""
        print(f'{bcolors.OKBLUE}{self.__class__.__name__}: '
              f'({self.__module__})\n'
              f'\tITP file is `{self.fname}`{bcolors.ENDC}')
        with open(self.fname, 'w', encoding="utf8") as f_w:
            f_w.write('; input pdb SMILES:\n')
            f_w.write('; Updated itp file for nanoparticle:\n')
            f_w.write('\n')
            self.write_molecule(f_w, itp.molecules)
            self.write_atoms(f_w, itp.atoms_updated)
            self.write_bonds(f_w, itp.bonds_updated)
            self.write_angles(f_w, itp.angles_updated)
            self.write_dihedrals(f_w, itp.dihedrals_updated)

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
        print(header)
        print(atoms)
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


if __name__ == '__main__':
    pass
