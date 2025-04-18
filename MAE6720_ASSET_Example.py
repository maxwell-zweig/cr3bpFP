import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

######
######
###### COMMENT: THIS CODE WORKS BUT DOES NOT HAVE FAILSAFE FOR TRY/CATCH. MUST ADD
######
######

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

###############################################################################

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
    phase.optimizer.set_PrintLevel(2)
    phase.optimize()

    return phase

def Full_Plot(Traj,IG,ref_state):

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    Traj_array = np.array(Traj).T
    IG_array = np.array(IG)

    if len(IG_array) == max(np.shape(IG_array)):
        IG_array = IG_array.T

    Thrust_Energy_Mag = DU/TU**2 * np.array([[np.sqrt(IG_array[7,i]**2 + IG_array[8,i]**2 + IG_array[9,i]**2)] for i in range(max(np.shape(IG_array)))])
    Thrust_Mass_Mag = DU/TU**2 * np.array([[np.sqrt(Traj_array[7,i]**2 + Traj_array[8,i]**2 + Traj_array[9,i]**2)] for i in range(max(np.shape(Traj_array)))])

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122, projection='3d')

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7])  # plots u_x vs time
    ax0.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[7], color='r')
    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8])  # plots u_y vs time
    ax1.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[8], color='r')
    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9])  # plots u_z vs time
    ax2.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[9], color='r')
    ax3.plot(TU/86400 * Traj_array[6], Thrust_Mass_Mag)     # plots u mag vs time
    ax3.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), Thrust_Energy_Mag, color='r')     # plots u mag vs time

    # plot the reference trajectory
    ax4.plot(ref_state[:,0], ref_state[:,1], ref_state[:,2], color=[0/255,0/255,0/255])
    # plot the energy optimal trajectory
    ax4.plot(IG_array[0], IG_array[1], IG_array[2], color='r')
    # plot the energy optimal trajectory
    ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2])
    # plot the moon
    ax4.scatter(1-mu_star,0,0, color=[130/255,130/255,130/255], s=20)

    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax0.set_ylabel(r'$U_x$')
    ax1.set_ylabel(r'$U_y$')
    ax2.set_ylabel(r'$U_z$')
    ax3.set_ylabel(r'$||U||$ [m/s$^2$]')
    ax3.set_xlabel(r't [days]')

    ax4.legend(["Reference", "Energy Optimal", "Mass Optimal"], loc="upper right")
    ax4.set_xlabel(r'$X$')
    ax4.set_ylabel(r'$Y$')
    ax4.set_zlabel(r'$Z$')
    fig.set_size_inches(10.5, 5.5, forward=True)

    fig.set_tight_layout(True)
    #plt.savefig('MassOptimal_Example2.png', dpi=500)
    #plt.savefig('MassOptimal_Example2.pdf')
    plt.show()

def Compare_Plot(Traj,IG,ref_state):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # plot the reference trajectory
    ax.plot(ref_state[:,0], ref_state[:,1], ref_state[:,2], color=[0,0,0])

    # plot the energy optimal trajectory
    IG_array = np.array(IG)
    ax.plot(IG_array[0,:], IG_array[1,:], IG_array[2,:], color=[109/255,0,234/255])

    # plot the mass optimal trajectory
    Traj_array = np.array(Traj).T
    ax.plot(Traj_array[0], Traj_array[1], Traj_array[2], color='b')

    ax.legend(["Reference", "Energy Optimal", "Mass Optimal"], loc="upper right")
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    fig.set_size_inches(10.5, 5.5, forward=True)

    fig.set_tight_layout(True)
    plt.show()

###############################################################################

if __name__ == "__main__":

    mu_star =  0.01215059   # Constant for CR3BP
    Umax = 0.0184           # DU/TU^2
    
    # REFERENCE TRAJECTORY STATE AND PERIOD
    tf  = 2.085034838884136 # TU
    # Load in the reference state data
    FileName_ref = "EnergyOptimal_state_EarthMoon_L2nrho.mat"
    ref_state = list(scipy.io.loadmat(FileName_ref).values())[-1]

    # Load in the energy-optimal state data for all runs (1000 orbits)
    FileName_state = "C:/Users/ccm41/Downloads/Research/Starlift/STM_Stuff/STMPrecompute-main/STMPrecompute-main/state_dataset_L2nrho_1000.mat"
    state_all = list(scipy.io.loadmat(FileName_state).values())[-1] #shape is (num_of_state_components, num_of_timesteps, num_of_runs)
    # Obtain data for a single energy optimal run

    ####################################################################
    ####################################################################
    ####################################################################
    ### CASE WHERE E-OPTIMAL AND T-OPTIMAL ARE DIFFERENT
    ####################################################################
    ####################################################################
    ####################################################################

    IG_state = state_all[:,:,212] #third index indicates which run it is from the dataset
    # IG takes the form of [state,time,control]
    #IG = [[IG_state[0,i], IG_state[1,i], IG_state[2,i], IG_state[3,i], IG_state[4,i], IG_state[5,i], tf*i/(max(np.shape(IG_state))), IG_state[9,i], IG_state[10,i], IG_state[11,i]] for i in range(max(np.shape(IG_state)))]
    IG = [[ref_state[i,0], ref_state[i,1], ref_state[i,2], ref_state[i,3], ref_state[i,4], ref_state[i,5], tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)

    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]
    optType = "LGL3"
    E_or_T = 0
    Umax = 0.0184           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 64
    numThreads = 8
    MeshTol = 1.0e-8
    EControl = 1.0e-10

    phase = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)


    Traj = phase.returnTraj()

    IG = Traj

    ####################################################################
    ####################################################################
    ####################################################################

    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]
    optType = "LGL3"
    E_or_T = 1
    Umax = 0.0184           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 100
    numThreads = 8
    MeshTol = 1.0e-10
    EControl = 1.0e-12

    phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
    Traj = phase2.returnTraj()

    ####################################################################
    ####################################################################
    ####################################################################

    Tab = phase2.returnTrajTable()

    integTab = ode.integrator(0.00001,Tab)
    integTab.setAbsTol(1.0e-16)
    integTab.setRelTol(1.0e-14)

    TrajI   = np.array(integTab.integrate_dense(Traj[0],tf))

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    deviation = np.reshape(np.array(IG_state[:6,0]),(1,6)) - np.array(ref_state[0,:6])
    # Below is one method to calculate dV
    Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
    # dV = sum(np.array([[(Thrust_Mag[i]+Thrust_Mag[i-1])/2*(TrajI[i,6]-TrajI[i-1,6])] for i in range(1,max(np.shape(TrajI)))])) # in DU/TU
    # dV = dV*DU/TU
    # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
    dV = DU/TU*scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1]

    if np.linalg.norm(TrajI[-1][:6] - TrajI[0][:6]) < MeshTol and max(Thrust_Mag) < Umax:
        f = 1
    else:
        f = 0


    ## Example of how to get exact timing statistics should you need to
    print("Deviation to Reference:      ",deviation)
    print("Delta V:                     ",dV," m/s")
    print("Total Runtime:               ",phase.optimizer.LastTotalTime + phase2.optimizer.LastTotalTime," s")
    print("Energy-Optimal Runtime:      ",phase.optimizer.LastTotalTime," s")
    print("Mass-Optimal Runtime:        ",phase2.optimizer.LastTotalTime," s")
                                                                             
    Full_Plot(TrajI,IG_state,ref_state)

    ###########################################################################








    ####################################################################
    ####################################################################
    ####################################################################
    ### CASE WHERE E-OPTIMAL AND T-OPTIMAL ARE SIMILAR
    ####################################################################
    ####################################################################
    ####################################################################

    IG_state = state_all[:,:,120] #third index indicates which run it is from the dataset
    # IG takes the form of [state,time,control]
    #IG = [[IG_state[0,i], IG_state[1,i], IG_state[2,i], IG_state[3,i], IG_state[4,i], IG_state[5,i], tf*i/(max(np.shape(IG_state))), IG_state[9,i], IG_state[10,i], IG_state[11,i]] for i in range(max(np.shape(IG_state)))]
    IG = [[ref_state[i,0], ref_state[i,1], ref_state[i,2], ref_state[i,3], ref_state[i,4], ref_state[i,5], tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)

    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]
    optType = "LGL3"
    E_or_T = 0
    Umax = 0.0184           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 64
    numThreads = 8
    MeshTol = 1.0e-8
    EControl = 1.0e-10

    phase = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)


    Traj = phase.returnTraj()

    IG = Traj

    ####################################################################
    ####################################################################
    ####################################################################


    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]
    optType = "LGL3"
    E_or_T = 1
    Umax = 0.0184           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 100
    numThreads = 8
    MeshTol = 1.0e-10
    EControl = 1.0e-12

    phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
    Traj = phase2.returnTraj()

    ####################################################################
    ####################################################################
    ####################################################################

    Tab = phase2.returnTrajTable()

    integTab = ode.integrator(0.00001,Tab)
    integTab.setAbsTol(1.0e-16)
    integTab.setRelTol(1.0e-14)

    TrajI   = np.array(integTab.integrate_dense(Traj[0],tf))

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    deviation = np.reshape(np.array(IG_state[:6,0]),(1,6)) - np.array(ref_state[0,:6])
    # Below is one method to calculate dV
    Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
    # dV = sum(np.array([[(Thrust_Mag[i]+Thrust_Mag[i-1])/2*(TrajI[i,6]-TrajI[i-1,6])] for i in range(1,max(np.shape(TrajI)))])) # in DU/TU
    # dV = dV*DU/TU
    # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
    dV = DU/TU*scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1]


    ## Example of how to get exact timing statistics should you need to
    print("Deviation to Reference:      ",deviation)
    print("Delta V:                     ",dV," m/s")
    print("Total Runtime:               ",phase.optimizer.LastTotalTime + phase2.optimizer.LastTotalTime," s")
    print("Energy-Optimal Runtime:      ",phase.optimizer.LastTotalTime," s")
    print("Mass-Optimal Runtime:        ",phase2.optimizer.LastTotalTime," s")
                                                                             
    Full_Plot(TrajI,IG,ref_state)

    ###########################################################################
    
    
    
    









    ####################################################################
    ####################################################################
    ####################################################################
    ### CASE WHERE E-OPTIMAL FAILS TO CONVERGE
    ####################################################################
    ####################################################################
    ####################################################################

    IG_state = state_all[:,:,212] #third index indicates which run it is from the dataset
    # IG takes the form of [state,time,control]
    #IG = [[IG_state[0,i], IG_state[1,i], IG_state[2,i], IG_state[3,i], IG_state[4,i], IG_state[5,i], tf*i/(max(np.shape(IG_state))), IG_state[9,i], IG_state[10,i], IG_state[11,i]] for i in range(max(np.shape(IG_state)))]
    IG = [[ref_state[i,0], ref_state[i,1], ref_state[i,2], ref_state[i,3], ref_state[i,4], ref_state[i,5], tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)

    BoundaryFirst = list(IG_state[:6,0]*10) + [0]
    BoundaryLast = list(IG_state[:6,0]*10) + [tf]
    optType = "LGL3"
    E_or_T = 0
    Umax = 0.0184*100           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 64
    numThreads = 8
    MeshTol = 1.0e-8
    EControl = 1.0e-10

    phase = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)


    Traj = phase.returnTraj()

    IG = Traj

    ####################################################################
    ####################################################################
    ####################################################################


    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]
    optType = "LGL7"
    E_or_T = 1
    Umax = 0.0184*100           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 200
    numThreads = 8
    MeshTol = 1.0e-10
    EControl = 1.0e-12

    phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
    Traj = phase2.returnTraj()

    ####################################################################
    ####################################################################
    ###################################################################
    Tab = phase2.returnTrajTable()

    integTab = ode.integrator(0.00001,Tab)
    integTab.setAbsTol(1.0e-16)
    integTab.setRelTol(1.0e-14)

    TrajI   = np.array(integTab.integrate_dense(Traj[0],tf))

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    deviation = np.reshape(np.array(IG_state[:6,0]),(1,6)) - np.array(ref_state[0,:6])
    # Below is one method to calculate dV
    Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
    # dV = sum(np.array([[(Thrust_Mag[i]+Thrust_Mag[i-1])/2*(TrajI[i,6]-TrajI[i-1,6])] for i in range(1,max(np.shape(TrajI)))])) # in DU/TU
    # dV = dV*DU/TU
    # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
    dV = DU/TU*scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1]


    ## Example of how to get exact timing statistics should you need to
    print("Deviation to Reference:      ",deviation)
    print("Delta V:                     ",dV," m/s")
    print("Total Runtime:               ",phase.optimizer.LastTotalTime + phase2.optimizer.LastTotalTime," s")
    print("Energy-Optimal Runtime:      ",phase.optimizer.LastTotalTime," s")
    print("Mass-Optimal Runtime:        ",phase2.optimizer.LastTotalTime," s")
                                                                             
    Full_Plot(TrajI,IG,ref_state)

    ###########################################################################















    ####################################################################
    ####################################################################
    ####################################################################
    ### CASE WHERE E-OPTIMAL AND T-OPTIMAL ARE HIGH-THRUST
    ####################################################################
    ####################################################################
    ####################################################################
    
    Umax = 0.0184*10000           # DU/TU^2
    
    IG = Traj
    optType = "LGL3"
    E_or_T = 0
    numKnots = 64
    numThreads = 8
    MeshTol = 1.0e-6
    EControl = 1.0e-7
    BoundaryFirst = list(IG_state[:6,0]) + [0]
    BoundaryLast = list(IG_state[:6,0]) + [tf]

    phase = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)

    phase2 = run_optimizer(ode, phase.returnTraj(), BoundaryFirst, BoundaryLast, "LGL7", 1, Umax, Umin, 250, 8, 1.0e-10, 1.0e-12)
    Traj = phase2.returnTraj()

    Tab = phase2.returnTrajTable()

    integTab = ode.integrator(0.00001,Tab)
    integTab.setAbsTol(1.0e-16)
    integTab.setRelTol(1.0e-14)

    TrajI   = np.array(integTab.integrate_dense(Traj[0],tf))

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    deviation = np.reshape(np.array(IG_state[:6,0]),(1,6)) - np.array(ref_state[0,:6])
    # Below is one method to calculate dV
    Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
    # dV = sum(np.array([[(Thrust_Mag[i]+Thrust_Mag[i-1])/2*(TrajI[i,6]-TrajI[i-1,6])] for i in range(1,max(np.shape(TrajI)))])) # in DU/TU
    # dV = dV*DU/TU
    # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
    dV = DU/TU*scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1]


    ## Example of how to get exact timing statistics should you need to
    print("Deviation to Reference:      ",deviation)
    print("Delta V:                     ",dV," m/s")
    print("Total Runtime:               ",phase.optimizer.LastTotalTime + phase2.optimizer.LastTotalTime," s")
    print("Energy-Optimal Runtime:      ",phase.optimizer.LastTotalTime," s")
    print("Mass-Optimal Runtime:        ",phase2.optimizer.LastTotalTime," s")
                                                                             
    Full_Plot(TrajI,IG,ref_state)

    ###########################################################################