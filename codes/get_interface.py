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


class GetSurface:
    """find water surface of the system"""

    info_msg: str = ''  # Message to pass for logging and writing in consoul

    def __init__(self,
                 residues_atoms: dict[str, pd.DataFrame]  # All atoms in ress
                 ) -> None:
        self.get_interface(residues_atoms)

    def get_interface(self,
                      residues_atoms: dict[str, pd.DataFrame]  # All atoms
                      ) -> None:
        """get the water surface"""
        aptes_com: np.ndarray  # Center of mass of NP
        aptes_r: np.float64  # Radius of NP
        aptes_com, aptes_r = self.__get_np_radius_com(residues_atoms['APT'])
        print(type(aptes_r))

    def __get_np_radius_com(self,
                            aptes: pd.DataFrame  # All the APTES atoms
                            ) -> tuple[np.ndarray, np.float64]:
        """find the radius of the NP, just find max and min of points in z-axis
        and their difference will be the radius."""
        # Calculate the center of mass
        aptes_com: np.ndarray = np.average(aptes[['x', 'y', 'z']], axis=0)
        max_z: np.float64 = np.max(aptes['z'])
        min_z: np.float64 = np.min(aptes['z'])
        return aptes_com, np.round(max_z-min_z, 3)


if __name__ == '__main__':
    print('This script runs within get_data.py module!\n')
