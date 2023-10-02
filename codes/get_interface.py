"""To determine the level of APTES protonation, locating the water's
surface is necessary. This requires inputting all nanoparticle data
(COR and APT) for accurate placement. As the unwrap structure cannot
be used, the script identifies the location of each water molecule
based on the location of the O atom in the water molecules.

To proceed, we rely on two assumptions: The water atoms situated above
the nanoparticles' highest point are considered to belong to the
bottom of the system as they will eventually fall from the dataframe.
Additionally, any water molecule within or around the nanoparticle,
enclosed by the smallest sphere, should not be considered part of the
interface and should be removed.
Only APTES information is used for the nanoparticle, since they have
most outward particles.
"""


import typing
import numpy as np
import pandas as pd
import logger
import write_gro_file as wrgro
from colors_text import TextColor as bcolors


class GetSurface:
    """find water surface of the system"""

    info_msg: str = 'Message from GetSurface:\n'  # Message for logging
    # Set in "get_water_surface" method:
    interface_z: np.float64  # Average place of the water suraface at interface
    interface_std: np.float64  # standard diviasion of the water suraface
    contact_angle: np.float64  # Final contact angle

    def __init__(self,
                 residues_atoms: dict[str, pd.DataFrame],  # All atoms in ress
                 aptes: str,  # Name of the APTES chains
                 log: logger.logging.Logger,  # To log info in it
                 write_debug: bool,  # If wanted to write the pdb file
                 ) -> None:
        self.write_debug: bool = write_debug
        self.get_interface(residues_atoms, aptes)
        self.__write_msg(log)
        self.info_msg = ''  # Empety the msg

    def get_interface(self,
                      residues_atoms: dict[str, pd.DataFrame],  # All atoms
                      aptes: str,  # Name of the APTES chains
                      ) -> None:
        """get the water surface"""
        aptes_com: np.ndarray  # Center of mass of NP
        aptes_r: np.float64  # Radius of NP
        self.info_msg += '\tUnit for GRO file is [nm] and for PDB is [A]\n'
        self.info_msg += f'\tNanoparticle: {aptes}\n'
        aptes_com, aptes_r = self.__get_np_radius_com(residues_atoms[aptes])
        self.get_water_surface(residues_atoms['SOL'], aptes_com, aptes_r)

    def get_water_surface(self,
                          waters: pd.DataFrame,  # all the water moles
                          aptes_com: np.ndarray,  # Center of mass of NP
                          aptes_r: np.float64  # Radius of NP
                          ) -> None:
        """find the water surface, and return mean place of the surface
        and its standard deviation"""
        z_hi: np.float64  # Treshhold for water molecules in in oil phase
        z_lo: np.float64  # Treshhold for water molecules in in water phase
        z_hi = aptes_com[2] + aptes_r
        z_lo = aptes_com[2] - aptes_r
        # Get water under the oil phase
        water_sec: pd.DataFrame = self.__water_under_oil(waters, z_hi)
        # Cuboidal box of water's oxysgens with a cylindrical void from NP
        cuboid_with_hole: pd.DataFrame = \
            self.__get_water_no_np(water_sec, aptes_com, aptes_r)
        water_surface: pd.DataFrame = \
            self.__get_surface_topology(cuboid_with_hole, z_lo)
        self.interface_z, self.interface_std = \
            self.__analyse_surface(water_surface)
        self.contact_angle = self.__get_contact_angle(aptes_com, aptes_r)

    def __get_contact_angle(self,
                            aptes_com: np.ndarray,  # Center of mass of NP
                            aptes_r: np.float64  # Radius of NP
                            ) -> np.float64:
        """calculate the contact angle of the nanoparticle, I use the
        Fig 5b of Mass paper (Joeri Smith, 2022), based on the mean
        of the interface"""
        h_depth: float  # Depth of NP in water
        h_depth = aptes_r + np.abs(aptes_com[2] - self.interface_z)
        contact_angle: np.float64 = np.arccos((h_depth/aptes_r) - 1)
        self.info_msg += f'\tThe contact angle is: {contact_angle} [rad]'
        self.info_msg += f', {np.rad2deg(contact_angle)} [deg]\n'
        return contact_angle

    def __analyse_surface(self,
                          water_surface: pd.DataFrame  # Water at surface df
                          ) -> tuple[np.float64, np.float64]:
        """analys surface and calculate the thickness of the surface,
        mean, higher, and lower bond
        """
        z_mean: np.float64 = np.mean(water_surface['z'])
        std_d: np.float64 = np.std(water_surface['z'])
        self.info_msg += f'\tThe mean place of water`s surface is: {z_mean}\n'
        self.info_msg += f'\tThe standard diviation of surface is: {std_d}\n'
        return z_mean, std_d

    def __get_surface_topology(self,
                               cuboid_with_hole: pd.DataFrame,  # water's O
                               z_lo: np.float64  # Treshhold for water res
                               ) -> pd.DataFrame:
        """get water surface topology from oxygens in the surface
        Using the grid meshes in the x and y directions, the water_com
        in each grid with the highest z value is returned.
        """
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        x_data: np.ndarray = np.array(cuboid_with_hole['x'])
        y_data: np.ndarray = np.array(cuboid_with_hole['y'])
        z_data: np.ndarray = np.array(cuboid_with_hole['z'])
        mesh_size: np.float64  # Size of the grid
        x_mesh, y_mesh, mesh_size = self.__get_grid_xy(x_data, y_data)
        max_z_index: list[int] = []  # Index of the max value at each grid
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                # Define the boundaries of the current mesh element
                x_min_mesh, x_max_mesh = x_mesh[i, j], x_mesh[i, j] + mesh_size
                y_min_mesh, y_max_mesh = y_mesh[i, j], y_mesh[i, j] + mesh_size

                # Select atoms within the current mesh element based on X and Y
                ind_in_mesh = np.where((x_data >= x_min_mesh) &
                                       (x_data < x_max_mesh) &
                                       (y_data >= y_min_mesh) &
                                       (y_data < y_max_mesh) &
                                       (z_data > z_lo))
                if len(ind_in_mesh[0]) > 0:
                    max_z = np.argmax(z_data[ind_in_mesh])
                    max_z_index.append(ind_in_mesh[0][max_z])
        water_surface: pd.DataFrame = cuboid_with_hole.iloc[max_z_index]
        if self.write_debug != 'None':
            wrgro.write_gromacs_gro(water_surface, 'water_surface_debug.gro')
        return water_surface

    def __get_water_no_np(self,
                          waters: pd.DataFrame,  # All waters under oil section
                          aptes_com: np.ndarray,  # Center of mass of NP
                          aptes_r: np.float64  # Radius of NP
                          ) -> pd.DataFrame:
        """drop water under the NP, the rest are those needed for getting
        the surface"""
        waters_oxy: pd.DataFrame  # Oxygens of water residues
        waters_oxy = waters[waters['atom_name'] == 'OH2']
        x_values = waters_oxy['x'].values
        y_values = waters_oxy['y'].values

        # Calculate the squared Euclidean distance between each point&origin
        distances: np.ndarray = np.sqrt((x_values - aptes_com[0]) ** 2 +
                                        (y_values - aptes_com[1]) ** 2)

        # Create a boolean mask indicating the rows to keep
        mask: np.ndarray = distances > aptes_r

        # Filter the dataframe using the mask
        df_filtered: pd.DataFrame = waters_oxy[mask]
        if self.write_debug != 'None':
            wrgro.write_gromacs_gro(df_filtered, 'o_waters_debug.gro')
        self.info_msg += '\tOnly oxygen atoms [OH2] are selected for the' + \
            ' looking for the surface.\n'
        return df_filtered

    @staticmethod
    def __water_under_oil(waters: pd.DataFrame,  # All water residues
                          z_lim: np.float64  # Highest point of NP
                          ) -> pd.DataFrame:
        """return waters in the water sections and drop ones above oil"""
        return waters[waters['z'] < z_lim]

    def __get_np_radius_com(self,
                            aptes: pd.DataFrame  # All the APTES atoms
                            ) -> tuple[np.ndarray, np.float64]:
        """find the radius of the NP, just find max and min of points in z-axis
        and their difference will be the radius * 2 ."""
        # Calculate the center of mass
        if self.write_debug != 'None':
            wrgro.write_gromacs_gro(aptes, 'APTES_debug.gro')
        aptes_com: np.ndarray = np.average(aptes[['x', 'y', 'z']], axis=0)
        max_z: np.float64 = np.max(aptes['z'])
        min_z: np.float64 = np.min(aptes['z'])
        aptes_r: np.float64 = np.round(max_z-min_z, 3) / 2
        self.info_msg += f'\tThe center of mass of NP is: {aptes_com}\n'
        self.info_msg += f'\tThe radius of NP is: {aptes_r} but {aptes_r*1.1}'
        self.info_msg += ' is used for getting the interface\n'
        return aptes_com, aptes_r*1.1

    @staticmethod
    def __get_grid_xy(x_data: np.ndarray,  # x component of the coms
                      y_data: np.ndarray,  # y component of the coms
                      ) -> tuple[np.ndarray, np.ndarray, np.float64]:
        """return the mesh grid for the xy of sol"""
        x_min: np.float64 = np.min(x_data)
        y_min: np.float64 = np.min(y_data)
        x_max: np.float64 = np.max(x_data)
        y_max: np.float64 = np.max(y_data)
        mesh_size: np.float64 = (x_max-x_min)/100.
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        x_mesh, y_mesh = np.meshgrid(
            np.arange(x_min, x_max + mesh_size, mesh_size),
            np.arange(y_min, y_max + mesh_size, mesh_size))
        return x_mesh, y_mesh, mesh_size

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetSurface.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class WrapperGetSurface:
    """class for getting all the nanoparticles"""

    interface_std: np.float64
    interface_z: np.float64
    info_msg: str = 'Message from WrapperGetSurface:\n'  # Message for logging

    def __init__(self,
                 residues_atoms: dict[str, pd.DataFrame],  # All atoms in ress
                 log: logger.logging.Logger,  # To log info in it
                 param: dict[str, typing.Any]  # If wanted to write pdb file
                 ) -> None:
        self.interface_std, self.interface_z = \
            self.get_all_surfaces(residues_atoms, log, param)
        self.__write_msg(log)

    def get_all_surfaces(self,
                         residues_atoms: dict[str, pd.DataFrame],
                         log: logger.logging.Logger,
                         param: dict[str, typing.Any]
                         ) -> tuple[np.float64, np.float64]:
        """get the situations of all the nanoparticles"""
        interface_z: np.float64
        interface_std: np.float64
        interface_z_lst: list[np.float64] = []
        interface_std_lst: list[np.float64] = []

        for aptes in param['aptes']:
            water_surface = \
                GetSurface(residues_atoms, aptes, log, param['DEBUG'])
            interface_z_lst.append(water_surface.interface_z)
            interface_std_lst.append(water_surface.interface_std)
        interface_std, interface_z = \
            self.set_attributes(interface_z_lst, interface_std_lst)
        self.info_msg += f'\tMean of interface_z: `{interface_z}`\n'
        self.info_msg += f'\tMean of interface_std: `{interface_std}`\n'
        return interface_std, interface_z

    @staticmethod
    def set_attributes(interface_z_lst: list[np.float64],
                       interface_std_lst: list[np.float64],
                       ) -> tuple[np.float64, np.float64]:
        """get average of the water surface"""
        interface_z: np.float64 = np.mean(interface_z_lst)
        interface_std: np.float64 = np.mean(interface_std_lst)
        return interface_std, interface_z


    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{WrapperGetSurface.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print('This script runs within get_data.py module!\n')
