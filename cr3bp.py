import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class CR3BP_Thrust_Dynamics(oc.ODEBase):

    def __init__(self,mu_star):
        Xvars = 6
        Uvars = 3
        ####################################################
        XtU = oc.ODEArguments(Xvars,Uvars)

        x,y,z,xdot,ydot,zdot = XtU.XVec().tolist()
        ux,uy,uz = XtU.UVec().tolist()

        R_31 = ((x+mu_star)**2 + y**2 + z**2)**(0.5)
        R_32 = ((x-1+mu_star)**2 + y**2 + z**2)**(0.5)

        xddot = 2*ydot + x - (1-mu_star)*(x+mu_star)/R_31**3 - mu_star*(x-1+mu_star)/R_32**3 + ux
        yddot = -2*xdot + y - (1-mu_star)*y/R_31**3 - mu_star*y/R_32**3 + uy
        zddot = -(1-mu_star)*z/R_31**3 - mu_star*z/R_32**3 + uz
    
        ode = vf.stack([xdot,ydot,zdot,xddot,yddot,zddot])
        ####################################################
        super().__init__(ode,Xvars,Uvars)


def run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl):
    phase = ode.phase(optType, IG, numKnots)
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,7), BoundaryFirst) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,7), BoundaryLast)
    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0
    #Produce a mass-optimal result by integrating over a norm() applied to the thrust vector
    if E_or_T == 0:
        phase.addIntegralObjective(Args(3).squared_norm(),[7,8,9]) 
    else:
        phase.addIntegralObjective(Args(3).norm(),[7,8,9])

    
    # here, we are adding adaptive meshing
    phase.setAdaptiveMesh(True)  #Enable Adaptive mesh for all following solve/optimize calls
    #phase.setMeshErrorEstimator('deboor')     #default
    ## Set Error tolerance on mesh (epsilon)
    phase.setMeshTol(MeshTol)  #default = 1.0e-6 
    ## Make sure to set optimizer EContol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(EControl)
    ## Set Max number of mesh iterations:
    #phase.setMaxMeshIters(10)  #default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(1)
    phase.optimize()

    return phase

if __name__ == "__main__":
    mu_star =  0.01215059   # Constant for CR3BP
    Umax = 0.0184*100           # DU/TU^2
    
    # REFERENCE TRAJECTORY STATE AND PERIOD
    tf  = 2.085034838884136 # TU
    # Load in the reference state data
    FileName_ref = "EnergyOptimal_state_EarthMoon_L2nrho.mat"
    ref_state = list(scipy.io.loadmat(FileName_ref).values())[-1]
    print(ref_state[0])

    IG = [[ref_state[i,0], ref_state[i,1], ref_state[i,2], ref_state[i,3], ref_state[i,4], ref_state[i,5], tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)
        

    optType = "LGL3"
    E_or_T = 0
    numKnots = 64
    numThreads = 8
    Umin = 1.0e-8
    MeshTol = 1.0e-8
    EControl = 1.0e-10
    BoundaryFirst = list(IG_state[:6,0]*10) + [0]
    BoundaryLast = list(IG_state[:6,0]*10) + [tf]

    phase = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)


    Traj = phase.returnTraj()

    IG = Traj

    ####################################################################
    ####################################################################
    ####################################################################

    optType = "LGL7"
    E_or_T = 1
    numKnots = 200
    numThreads = 8
    MeshTol = 1.0e-10
    EControl = 1.0e-12
    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]

    phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
    Traj = phase2.returnTraj()
    