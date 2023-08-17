"""reading itp files and return data several data frames
    Usually, the itp file contains information about atoms (not
    coordinates), bonds, angles, dihedrals, and improper dihedrals.

    informations in the itp file:
        [ moleculetype ] : defines the name of your molecule in this top
    and nrexcl = 3 stands for excluding non-bonded interactions between
    atoms that are no further than 3 bonds away.

        [ atoms ] : defines the molecule, where nr and type are fixed,
    the rest is user defined. So atom can be named as you like,
    cgnr made larger or smaller (if possible, the total charge of
    a charge group should be zero), and charges can be changed here
    too.

        [ bonds ] : no comment.

        [ pairs ] : LJ and Coulomb 1-4 interactions

        [ angles ] : no comment

        [ dihedrals ] : in this case there are 9 proper dihedrals
    (funct = 1), 3 improper (funct = 4) and no Ryckaert-Bellemans type
    dihedrals.
    """

import sys
import typing
import pandas as pd
from colors_text import TextColor as bcolors


# A helper function needed by most of the classes to clean the lines
def free_char_line(line: str  # line of the itp file
                   ) -> list[str]:  # Free from chars
    """cheack the lines and return the line free special chars"""
    char_list: list[str] = [';', '#', ':', '...']  # chars eliminate in lines
    l_line: list[str]  # Breaking the line cahrs
    l_line = line.strip().split(' ')
    l_line = [item for item in l_line if item]
    l_line = [item for item in l_line if item not in char_list]
    return l_line


class Itp:
    """read itp file and return a DataFrame of the information
    within the file"""
    def __init__(self,
                 fname: str  # Name of the itp file
                 ) -> None:
        print(f"{bcolors.OKBLUE}Reading '{fname}' ...{bcolors.ENDC}")
        self.get_itp(fname)

    def get_itp(self,
                fname: str  # Name of the itp file
                ) -> None:
        """read the file line by line and call related methods"""
        atoms: bool = False  # Flag of 'atoms' occurrence
        bonds: bool = False  # Flag of 'bonds' occurrence
        angles: bool = False  # Flag of 'angles' occurrence
        dihedrals: bool = False  # Flag of 'dihedrals' occurrence
        impropers: bool = False  # Flag of 'impropers' occurrence
        moleculetype: bool = False  # Flag of 'moleculetype' occurrence
        atoms_info: list[str] = []  # to append atoms lines
        bonds_info: list[str] = []  # to append bonds lines
        angles_info: list[str] = []  # to append angles lines
        dihedrals_info: list[str] = []  # to append dihedrals lines
        moleculetype_info: list[str] = []  # to append dihedrals lines
        with open(fname, 'r', encoding='utf8') as f_r:
            while True:
                line: str = f_r.readline()
                if line.strip():
                    if line.strip().startswith('['):
                        wilds: list[str]  # Parts of the line
                        wilds = line.strip().split()
                        if wilds[1] == 'atoms':
                            atoms, bonds, angles, dihedrals, impropers,\
                                moleculetype = True, False, False, False,\
                                False, False
                        elif wilds[1] == 'bonds':
                            atoms, bonds, angles, dihedrals, impropers,\
                                moleculetype = False, True, False, False,\
                                False, False
                        elif wilds[1] == 'angles':
                            atoms, bonds, angles, dihedrals, impropers,\
                                moleculetype = False, False, True, False,\
                                False, False
                        elif wilds[1] == 'dihedrals':
                            atoms, bonds, angles, dihedrals, impropers,\
                                moleculetype = False, False, False, True,\
                                False, False
                        elif wilds[1] == 'moleculetype':
                            atoms, bonds, angles, dihedrals, impropers,\
                                moleculetype = False, False, False, False,\
                                False, True
                        else:
                            atoms, bonds, angles, dihedrals, impropers,\
                                moleculetype = False, False, False, False,\
                                False, False
                    else:
                        if moleculetype:
                            moleculetype_info.append(line)
                        if atoms:
                            atoms_info.append(line)
                        if bonds:
                            bonds_info.append(line)
                        if angles:
                            angles_info.append(line)
                        if dihedrals:
                            if 'impropers' not in free_char_line(line):
                                impropers = False
                            else:
                                impropers = True
                                dihedrals = False
                        if dihedrals and not impropers:
                            dihedrals_info.append(line)
                if not line:
                    break
        atom = AtomsInfo(atoms_info)
        molecules = MoleculeInfo(moleculetype_info)
        bond = BondsInfo(atoms=atom.df, bonds=bonds_info)
        angle = AnglesInfo(atoms=atom.df, angles=angles_info)
        dihedral = DihedralsInfo(atoms=atom.df, dihedrals=dihedrals_info)
        self.atoms = atom.df
        self.bonds = bond.df
        self.angles = angle.df
        self.dihedrals = dihedral.df
        self.molecules = molecules.df


class MoleculeInfo:
    """get molecules wild information and return the line and its info"""
    def __init__(self,
                 molecules: list[str]  # line read by Itp class
                 ) -> None:
        self.get_molecule_info(molecules)

    def get_molecule_info(self,
                          molecules: list[str]  # line about molecules
                          ) -> None:
        """read and return data about molecule"""
        l_line: list[str]  # Breaking the line chars
        columns: list[str] = ['Name', 'nrexcl']
        name: list[str] = []  # Name of the molecules
        style: list[str] = []
        for line in molecules:
            tmp = line.strip().split('\t')
            tmp = [item for item in tmp if tmp]
            line = ' '.join(tmp)
            if line.startswith(';'):
                l_line = free_char_line(line)
                if l_line != columns:
                    sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                             f'\tError in the [ atoms ] header of the '
                             f'itp file\n{bcolors.ENDC}')
            else:
                l_line = line.split(' ')
                l_line = [item for item in l_line if item]
                name.append(l_line[0])
                style.append(l_line[1])
        self.df = self.mk_df(name, style, columns)

    @staticmethod
    def mk_df(name: list[str],  # Name of the molecules
              style: list[str],  # Style of the interactions(!)
              columns: list[str]  # Name of the columns
              ) -> pd.DataFrame:
        """make a dataframe to save all the data"""
        df_molecules = pd.DataFrame(columns=columns)
        df_molecules['Name'] = name
        df_molecules['nrexcl'] = style
        return df_molecules


class AtomsInfo:
    """get atoms wild information and retrun a DataFrame"""
    def __init__(self,
                 atoms: list[str]  # lines read by Itp class
                 ) -> None:
        self.df = self.get_atoms_info(atoms)

    def get_atoms_info(self,
                       atoms: list[str]  # Lines of the atoms' section
                       ) -> pd.DataFrame:
        """get atoms info from the file"""
        l_line: list[str]  # Breaking the line chars
        # Check if header of the atoms section is same as the defeined one
        columns: list[str]   # columns for the atoms dict, name of each column
        columns = ['atomnr', 'atomtype', 'resnr', 'resname', 'atomname',
                   'chargegrp', 'charge', 'mass', 'element']
        atomnr: list[typing.Any] = []  # list to append info: atoms id
        atomtype: list[typing.Any] = []  # list to append info: forcefield type
        resnr: list[typing.Any] = []  # list to append info: res infos
        resname: list[typing.Any] = []  # list to append info: res number
        atomname: list[typing.Any] = []  # list to append info: atom name
        chargegrp: list[typing.Any] = []  # list to append info: charge group
        charge: list[typing.Any] = []  # list to append info: charge value
        mass: list[typing.Any] = []  # list to append info: mass value
        element: list[typing.Any] = []  # list to append info: real name
        for line in atoms:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'Total' not in l_line:  # Not header!
                    if l_line != columns:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ atoms ] header of the '
                                 f'itp file\n{bcolors.ENDC}')
            else:
                l_line = free_char_line(line)
                atomnr.append(l_line[0])
                atomtype.append(l_line[1])
                resnr.append(l_line[2])
                resname.append(l_line[3])
                atomname.append(l_line[4])
                chargegrp.append(l_line[5])
                charge.append(l_line[6])
                mass.append(l_line[7])
                element.append(l_line[8])
        df_atoms: pd.DataFrame  # DataFrame from the infos
        df_atoms = pd.DataFrame(columns=columns)
        df_atoms['atomnr'] = atomnr
        df_atoms['atomtype'] = atomtype
        df_atoms['resnr'] = resnr
        df_atoms['resname'] = resname
        df_atoms['atomname'] = atomname
        df_atoms['chargegrp'] = chargegrp
        df_atoms['charge'] = charge
        df_atoms['mass'] = mass
        df_atoms['element'] = self.drop_dot(element)
        return df_atoms

    @staticmethod
    def drop_dot(chars: list[str]  # to drop . from its items
                 ) -> list[str]:
        """drop dot (.) from the string"""
        return [item.replace('.', '') for item in chars]


class BondsInfo:
    """get the bonds list from Itp class and return a dataframe"""
    def __init__(self,
                 bonds: list[str],  # lines of bonds section read by Itp class
                 atoms: pd.DataFrame  # atoms df from AtomsInfo to get names
                 ) -> None:
        """get the bonds infos"""
        self.mk_bonds_df(bonds, atoms)

    def mk_bonds_df(self,
                    bonds: list[str],  # lines of bonds section
                    atoms: pd.DataFrame  # atoms df from AtomInfo
                    ) -> None:
        """call all the methods to make the bonds DataFrame"""
        a_i: list[int]  # index of the 1st atoms in the bonds
        a_j: list[int]  # index of the 2nd atoms in the bonds
        funct: list[int]  # index of the type of the bonds
        names: list[str]  # name of the bonds
        a_i, a_j, funct, names = self.get_bonds(bonds)
        self.df = self.mk_df(a_i, a_j, funct, names, atoms)

    def get_bonds(self,
                  bonds: list[str]  # lines of bonds section read by Itp class
                  ) -> tuple[list[int], list[int], list[int], list[str]]:
        """return bonds dataframe to make bonds dataframe"""
        columns: list[str]  # Columns of the bonds wild
        columns = ['ai', 'aj', 'typ', 'cmt', 'name']
        a_i: list[int] = []  # index of the 1st atoms in the bonds
        a_j: list[int] = []  # index of the 2nd atoms in the bonds
        funct: list[int] = []  # Type of the function of the bond
        names: list[str] = []  # name of the bonds
        for line in bonds:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'Total' not in l_line:  # Not header!
                    if l_line != columns:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ bonds ] header of the '
                                 f'itp file\n{bcolors.ENDC}')
            else:
                tmp = line.split()
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(int(l_line[0]))
                a_j.append(int(l_line[1]))
                funct.append(int(l_line[2]))
                names.append(l_line[3])
        return a_i, a_j, funct, names

    def mk_df(self,
              a_i: list[int],  # index of the 1st atom in the bonds
              a_j: list[int],  # index of the 2nd atom in the bonds
              funct: list[int],  # Index of the type of the bond
              names: list[str],  # names of the bonds form bonds section
              atoms: pd.DataFrame  # atoms df from AtomsInfo to cehck the name
              ) -> pd.DataFrame:  # bonds DataFrame
        """make DataFrame and check if they are same as atoms name"""
        df_bonds: pd.DataFrame  # to save the bonds_df
        df_bonds = pd.DataFrame(columns=['ai', 'aj', 'typ', 'cmt', 'name'])
        df_bonds['ai'] = a_i
        df_bonds['aj'] = a_j
        df_bonds['name'] = names
        df_bonds['cmt'] = [';' for _ in a_i]
        df_bonds['typ'] = funct
        df_bonds = self.check_names(df_bonds, atoms)
        df_bonds.index += 1
        return df_bonds

    @staticmethod
    def check_names(df_bonds: pd.DataFrame,  # Df to check its names column
                    atoms: pd.DataFrame  # Df source (mostly atoms df)
                    ) -> None:
        """checks the names column in the source file with comparing it
        with names from the atoms dataframe name column for each atom"""
        df_c: pd.DataFrame = df_bonds.copy()
        ai_name: list[str]  # name of the 1st atom from the list
        aj_name: list[str]  # name of the 2nd atom from the list
        ai_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_bonds['ai']]
        aj_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_bonds['aj']]
        names: list[str] = [f'{i}-{j}' for i, j in zip(ai_name, aj_name)]
        df_c['name'] = names
        return df_c


class AnglesInfo:
    """get the angles list from Itp class and return a dataframe"""
    def __init__(self,
                 angles: list[str],  # lines of angles section by Itp class
                 atoms: pd.DataFrame  # atoms df from AtomsInfo to get names
                 ) -> None:
        """get the angles infos"""
        self.mk_angles_df(angles, atoms)

    def mk_angles_df(self,
                     angles: list[str],  # lines of angles section
                     atoms: pd.DataFrame  # atoms df from AtomInfo
                     ) -> None:
        """call all the methods to make the bonds DataFrame"""
        a_i: list[int]  # index of the 1st atoms in the angles
        a_j: list[int]  # index of the 2nd atoms in the angles
        a_k: list[int]  # index of the 3rd atoms in the angles
        names: list[str]  # name of the angles
        a_i, a_j, a_k, funct, names = self.get_angles(angles)
        self.df = self.mk_df(a_i, a_j, a_k, funct, names, atoms)

    def get_angles(self,
                   angles: list[str],  # lines of angles section by Itp class
                   ) -> tuple[list[int], list[int], list[int],
                              list[int], list[str]]:
        """return bonds dataframe to make angles dataframe"""
        columns: list[str]  # Columns of the angles wild
        columns = ['ai', 'aj', 'ak', 'typ', 'cmt', 'name']
        a_i: list[int] = []  # index of the 1st atoms in the angles
        a_j: list[int] = []  # index of the 2nd atoms in the angles
        a_k: list[int] = []  # index of the 3rd atoms in the angles
        funct: list[int] = []  # index of the type (function) the angles
        names: list[str] = []  # name of the angles

        for line in angles:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'Total' not in l_line:  # Not header!
                    if l_line == columns:
                        pass
                    else:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ angles ] header of the '
                                 f'itp file\n{bcolors.ENDC}')
            else:
                tmp = line.strip().split('\t')
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(int(l_line[0]))
                a_j.append(int(l_line[1]))
                a_k.append(int(l_line[2]))
                funct.append(int(l_line[3]))
                names.append(l_line[4])
        return a_i, a_j, a_k, funct, names

    def mk_df(self,
              a_i: list[int],  # index of the 1st atom in the angles
              a_j: list[int],  # index of the 2nd atom in the angles
              a_k: list[int],  # index of the 3rd atom in the angles
              funct: list[int],  # Type (function) of the angles
              names: list[str],  # names of the bonds form angles section
              atoms: pd.DataFrame  # atoms df from AtomsInfo to cehck the name
              ) -> pd.DataFrame:  # angles DataFrame
        """make DataFrame and check if they are same as atoms name"""
        df_angles: pd.DataFrame  # to save the angles_df
        df_angles = \
            pd.DataFrame(columns=['ai', 'aj', 'ak', 'typ', 'cmt', 'name'])
        df_angles['ai'] = a_i
        df_angles['aj'] = a_j
        df_angles['ak'] = a_k
        df_angles['name'] = names
        df_angles['cmt'] = [';' for _ in a_i]
        df_angles['typ'] = funct
        df_angles = self.check_names(df_angles, atoms)
        df_angles.index += 1
        return df_angles

    @staticmethod
    def check_names(df_angles: pd.DataFrame,  # Df to check its names column
                    atoms: pd.DataFrame  # Df source (mostly atoms df)
                    ) -> None:
        """ checks the names column in the source file with comparing it
        with names from the atoms dataframe name column for each atom"""
        df_c: pd.DataFrame = df_angles.copy()
        ai_name: list[str]  # name of the 1st atom from the list
        aj_name: list[str]  # name of the 2nd atom from the list
        ak_name: list[str]  # name of the 3rd atom from the list
        ai_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_angles['ai']]
        aj_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_angles['aj']]
        ak_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_angles['ak']]
        names: list[str] = [f'{i}-{j}-{k}' for i, j, k
                            in zip(ai_name, aj_name, ak_name)]
        df_c['name'] = names
        return df_c


class DihedralsInfo:
    """get the dihdrals list from Itp class and return a dataframe"""
    def __init__(self,
                 dihedrals: list[str],  # lines of dihedrals section by Itp
                 atoms: pd.DataFrame  # atoms df from AtomsInfo to get names
                 ) -> None:
        """get the dihedrals infos"""
        self.mk_dihedrals_df(dihedrals, atoms)

    def mk_dihedrals_df(self,
                        dihedrals: list[str],  # lines of dihedrals section
                        atoms: pd.DataFrame  # atoms df from AtomInfo
                        ) -> None:
        """call all the methods to make the bonds DataFrame"""
        a_i: list[int]  # index of the 1st atoms in the dihedrals
        a_j: list[int]  # index of the 2nd atoms in the dihedrals
        a_k: list[int]  # index of the 3rd atoms in the dihedrals
        a_h: list[int]  # index of the 4th atoms in the dihedrals
        funct: list[int]  # index of the type of the dihedrals
        names: list[str]  # name of the dihedrals
        a_i, a_j, a_k, a_h, funct, names = self.get_dihedrals(dihedrals)
        self.df = self.mk_df(a_i, a_j, a_k, a_h, funct, names, atoms)

    def get_dihedrals(self,
                      dihedrals: list[str],  # lines of dihedrals section
                      ) -> pd.DataFrame:  # DataFrame of the dihedrals
        """return bonds dataframe to make dihedrals dataframe"""
        columns: list[str]  # Columns of the dihedrals wild
        columns = ['ai', 'aj', 'ak', 'ah', 'typ', 'cmt', 'name']
        a_i: list[int] = []  # index of the 1st atoms in the dihedrals
        a_j: list[int] = []  # index of the 2nd atoms in the dihedrals
        a_k: list[int] = []  # index of the 3rd atoms in the dihedrals
        a_h: list[int] = []  # index of the 4th atoms in the dihedrals
        funct: list[int] = []  # index of the type of the dihedrals
        names: list[str] = []  # name of the dihedrals
        for line in dihedrals:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'Total' not in l_line:  # Not header!
                    if l_line == columns:
                        pass
                    else:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ dihedrals ] header of the'
                                 f' itp file\n{bcolors.ENDC}')
            else:
                tmp = line.strip().split('\t')
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(int(l_line[0]))
                a_j.append(int(l_line[1]))
                a_k.append(int(l_line[2]))
                a_h.append(int(l_line[3]))
                funct.append(int(l_line[4]))
                names.append(l_line[5])
        return a_i, a_j, a_k, a_h, funct, names

    def mk_df(self,
              a_i: list[int],  # index of the 1st atom in the dihedrals
              a_j: list[int],  # index of the 2nd atom in the dihedrals
              a_k: list[int],  # index of the 3rd atom in the dihedrals
              a_h: list[int],  # index of the 4th atom in the dihedrals
              funct: list[int],  # types of the function
              names: list[str],  # names form dihedrals section
              atoms: pd.DataFrame  # atoms df from AtomsInfo to cehck the name
              ) -> pd.DataFrame:  # dihedrals DataFrame
        """make DataFrame and check if they are same as atoms name"""
        df_dihedrals: pd.DataFrame  # to save the dihedrals_df
        df_dihedrals = pd.DataFrame(
            columns=['ai', 'aj', 'ak', 'ah', 'typ', 'cmt', 'name'])
        df_dihedrals['ai'] = a_i
        df_dihedrals['aj'] = a_j
        df_dihedrals['ak'] = a_k
        df_dihedrals['ah'] = a_h
        df_dihedrals['name'] = names
        df_dihedrals['cmt'] = [';' for _ in a_i]
        df_dihedrals['typ'] = funct
        df_dihedrals = self.check_names(df_dihedrals, atoms)
        df_dihedrals.index += 1
        return df_dihedrals

    @staticmethod
    def check_names(df_dihedrals: pd.DataFrame,  # Df to check its names column
                    atoms: pd.DataFrame  # Df source (mostly atoms df)
                    ) -> None:
        """ checks the names column in the source file with comparing it
        with names from the atoms dataframe name column for each atom"""
        df_c: pd.DataFrame = df_dihedrals.copy()
        ai_name: list[str]  # name of the 1st atom from the list
        aj_name: list[str]  # name of the 2nd atom from the list
        ak_name: list[str]  # name of the 3rd atom from the list
        ah_name: list[str]  # name of the 4th atom from the list
        ai_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_dihedrals['ai']]
        aj_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_dihedrals['aj']]
        ak_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_dihedrals['ak']]
        ah_name = [atoms.loc[atoms['atomnr'] == str(item)]['atomname'][item-1]
                   for item in df_dihedrals['ah']]
        names: list[str] = [f'{i}-{j}-{k}-{h}' for i, j, k, h
                            in zip(ai_name, aj_name, ak_name, ah_name)]
        df_c['name'] = names
        return df_c


if __name__ == '__main__':
    itp = Itp(sys.argv[1])
