# SAMAV2_2

Sight-seeing order of Macao World Heritage Sites

Case 1: Helicopter-style (Santa Claus and Traveling Salesman Problem)
        The optimal route is found based on the Simulated Annealing and Metropolis Algorithm.

Version 2_2: Change r in Version 1 to rCoor and r, where 
             rCoor refers to the real coordinates of latitude and longitude while
             r is the normalized coordinates for plotting.

Modification in Version 2_2: 1. distance directly used in km
                                          2. the animation update is modified

Comments: Path Plot is ok after modifying Lines 192 and 193,
                  replacing sites.X with sites.radiansX.