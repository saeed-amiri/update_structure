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


import numpy as np
import pandas as pd
import write_pdb_file as wrpdb


class GetSurface:
    """find water surface of the system"""

    info_msg: str = '-message:\n'  # Message to pass for logging and writing

    def __init__(self,
                 residues_atoms: dict[str, pd.DataFrame],  # All atoms in ress
                 write_debug: bool = False,  # If wanted to write the pdb file
                 ) -> None:
        self.write_debug: bool = write_debug
        self.get_interface(residues_atoms)

    def get_interface(self,
                      residues_atoms: dict[str, pd.DataFrame]  # All atoms
                      ) -> None:
        """get the water surface"""
        aptes_com: np.ndarray  # Center of mass of NP
        aptes_r: np.float64  # Radius of NP
        aptes_com, aptes_r = self.__get_np_radius_com(residues_atoms['APT'])
        self.get_water_surface(residues_atoms['SOL'], aptes_com, aptes_r)

    def get_water_surface(self,
                          waters: pd.DataFrame,  # all the water moles
                          aptes_com: np.ndarray,  # Center of mass of NP
                          aptes_r: np.float64  # Radius of NP
                          ) -> None:
        """find the water surface"""
        z_lim: np.float64  # Treshhold for water molecules
        z_lim = aptes_com[2] + aptes_r
        # Get water under the oil phase
        water_sec: pd.DataFrame = self.__water_under_oil(waters, z_lim)
        self.__get_water_no_np(water_sec, aptes_com, aptes_r)

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
        if self.write_debug: 
            wrpdb.WriteResiduePdb(df_filtered, 'o_waters.pdb')
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
        if self.write_debug: 
            wrpdb.WriteResiduePdb(aptes, 'APTES.pdb')
        aptes_com: np.ndarray = np.average(aptes[['x', 'y', 'z']], axis=0)
        max_z: np.float64 = np.max(aptes['z'])
        min_z: np.float64 = np.min(aptes['z'])
        aptes_r: np.float64 = np.round(max_z-min_z, 3) / 2 * 1.1
        self.info_msg += f'\tThe center of mass of NP is: {aptes_com}\n'
        self.info_msg += f'\tThe radius of NP is: {aptes_r}\n'
        return aptes_com, aptes_r


if __name__ == '__main__':
    print('This script runs within get_data.py module!\n')
