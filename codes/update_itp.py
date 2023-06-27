"""The script updates the "APT_COR.itp" file using the HN3 atoms set by
update_residue.py. To do this, we need to add the correct information
for the name and charge for the new HN3 in the atoms section.
Additionally, for chains that have been protonated, It also updates
the name and charges for N, HN1, and HN2 to match the new values.
The script needs to add bonds between N and the new HN3. Since force-
field values are set in different files, we only need to update the
sections for the extra H in each chain, including angles and dihedrals.
"""

import pandas as pd
import itp_to_df as itp


class UpdateItp(itp.Itp):
    """updating the itp file for each section"""
    # object from itp class
    # atoms: pd.DataFrame  # Section of initial itp file
    # bonds: pd.DataFrame  # Section of initial itp file
    # angles: pd.DataFrame  # Section of initial itp file
    # dihedrals: pd.DataFrame  # Section of initial itp file
    # molecules: pd.DataFrame  # Section of initial itp file

    def __init__(self,
                 fname: str,  # Name of the itp file
                 hn3: pd.DataFrame  # Information for new NH3 atoms
                 ) -> None:
        super().__init__(fname)
        self.update_itp(hn3)

    def update_itp(self,
                   hn3: pd.DataFrame  # NH3 atoms
                   ) -> None:
        """get all the data and return final df"""


if __name__ == '__main__':
    print("This script should be call from 'updata_pdb_itp.py")
