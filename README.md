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
            All the distance values are in angestrom and angles in degree
            ############### CONSTANT VALUES ##############
            The contact angle of the nanoparticle (NP)
            @ANGLE=90
            Estimate radius of the NP before functionalization with APTES
            @RADIUS=25
            ############### INTERACTIVE VALUES ###########
            intereface location -> caculated by analysing script
            If PDB then read it from the PDB file, otherwise from the fixed value set in here
            @PDB=True
            If no PDB the interface is got from here: (They should set anyway)
            @INTERFACE=12.81
            interface thickness
            @INTERFACE_WIDTH=10.47
            How to select atoms, under intereface: INTERFACE
            under upper bound of the interface: UPPERBOUND
            under lower bound of the interface: LOWERBOUND
            within the interface: WHITIN
            @LINE=UPPERBOUND
            the center of mass of the nanoparticle
            @NP_ZLOC=100
            ############### Computation parameters ###########
            Number of points to make to try to put H of new protonation there
            @NUMSAMPLE=100
            Distance to put between ION and other atoms
            @ION_DISTANCE=1
            Number of times try to find a place for an ion
            @ION_ATTEPTS=100
    -itp files:
        It needs itp file for the following:
            -protonated APTES,
            -unprotonated APTES,
            -counterion
