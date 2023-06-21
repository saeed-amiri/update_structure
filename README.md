# update_structure
This program examines the pdb file to verify if the NP is positioned
below the interface. If there are any unprotonated (protonated) APTES
molecules in the water (oil) phase, it will modify the APTES chain by
adding (removing) an H atom to the system. It may also change the names
of the remaining particles in the chain as required and, most
importantly, adjust the charges of all the particles in the chain.
Depending on the alterations in the system, it will update the number
of counterions to ensure that the system stays neutral. To operate
correctly, the program requires the interface located at the beginning
of the simulation for comparison purposes and the ability to change
the NP's location.

Inputs:
    -input.pdb:
        Structure of the data file.
        It must be pdb.
    -update_param:
        this file contains information about interface and nanoparticle
        and has a structure as follows:
        # All the distance values are in angstroms and angles in degree
        ############### CONSTANT VALUES ##############
        # The contact angle of the nanoparticle (NP)
        ANGLE=90
        # Estimated radius of the NP before functionalization with APTES
        RADIUS=25
        ############### INTERACTIVE VALUES ###########
        # Intereface location -> calculated by analyzing-script
        INTERFACE=12
        # interface thickness
        INTERFACE_WIDTH=10
        ############### Computation parameters ###########
        Number of points to make to try to put H of new protonation there
        @NUMSAMPLE=100
    -itp files:
        It needs itp file for the following:
            -protonated APTES,
            -unprotonated APTES,
            -counterion
