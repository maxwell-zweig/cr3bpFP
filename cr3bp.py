import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

   
DU = 384400000.0000000
TU = 2.360584684800000E+06/(2*np.pi)

mu_star =  0.01215059   # Constant for CR3BP
Umax = 0.0184*100  # DU/TU^2

# REFERENCE TRAJECTORY STATE AND PERIOD
tf  = 2.085034838884136 # TU
# Load in the reference state data
FileName_ref = "EnergyOptimal_state_EarthMoon_L2nrho.mat"
ref_state = list(scipy.io.loadmat(FileName_ref).values())[-1]

Umin = 1.0e-8

optType = "LGL7"
E_or_T = 1
numKnots = 200
numThreads = 8
MeshTol = 1.0e-10
EControl = 1.0e-12


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


def FrontBackEqCon():
    X_0, t_0, X_f, t_f = Args(14).tolist([(0,6), (6,1), (7,6), (13,1)])
    eq1 = X_0[3:] - X_f[3:]
    return eq1


def plot_traj(Traj_array):
    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(426)

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7])  # plots u_x vs time

    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8])  # plots u_y vs time

    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9])  # plots u_z vs time

    ax3.plot(TU / 86400 * Traj_array[6], DU / TU ** 2 * np.linalg.norm(Traj_array[7:10], axis=0))


    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)  


    ax0.set_ylabel(r'$U_x$')
    ax1.set_ylabel(r'$U_y$')
    ax2.set_ylabel(r'$U_z$')
    ax3.set_ylabel(r'$Control Magnitude$')

    fig.set_size_inches(10.5, 5.5, forward=True)


    fig.set_tight_layout(True)
    #plt.savefig('MassOptimal_Example2.png', dpi=500)
    #plt.savefig('MassOptimal_Example2.pdf')
    plt.show()



def compute_trajectories():
    print("running pole sitter computation")
    target_x = 1 - mu_star
    target_y = 0
    target_z = np.linspace(0, 0.5, 20)

    IG = [[ref_state[i,0] * 3, ref_state[i,1] * 3, ref_state[i,2] * 3, ref_state[i,3] * 3, ref_state[i,4] * 3, ref_state[i,5] * 3, tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)

    for _, targ in enumerate(target_z):


        BoundaryFirst = list([target_x, target_y, targ]) + [0]
        BoundaryLast =  list([target_x, target_y, targ]) + [tf]

        phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
        Traj = phase2.returnTraj()
        Full_Plot(Traj,ref_state,ref_state)
        



def run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl):
    phase = ode.phase(optType, IG, numKnots)
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,3), BoundaryFirst[0:3]) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,3), BoundaryLast[0:3])

    # adding the time boundary conditions
    phase.addBoundaryValue("First", [6], BoundaryFirst[3])
    phase.addBoundaryValue("Last", [6], BoundaryLast[3])


    # enforce perioodicity
    phase.addEqualCon("FirstandLast", FrontBackEqCon(), range(0,7), [], [])


    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0
    #Produce a mass-optimal result by integrating over a norm() applied to the thrust vector
    if E_or_T == 0:
        phase.addIntegralObjective(Args(3).squared_norm(),[7,8,9]) 
    else:
        phase.addIntegralObjective(Args(3).norm(),[7,8,9])


    XtUVars = [2]
    OPVars = []
    SPVars = []


    def inequalCon():
        x3 = Args(1).tolist()
        return x3

 
    # bound the object below the line z = 0 
    PhaseRegion = "Path"
    VarIndex = 3 
    upperBound = 0 
    scale = 1
   # phase.addUpperVarBound(PhaseRegion, VarIndex, upperBound, scale)

    
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
 

    IG = [[ref_state[i,0] * 3, ref_state[i,1] * 3, ref_state[i,2] * 3, ref_state[i,3] * 3, ref_state[i,4] * 3, ref_state[i,5] * 3, tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)

   

    #####################################################################
    #####################################################################
    ##############################e######################################

    
    '''
    BoundaryFirst = list(ref_state[0, 0 : 3] * 1.1) + [0]
    BoundaryLast =  list(ref_state[0, 0 : 3] * 1.1) + [tf]

    phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
    Traj = phase2.returnTraj()

    traj = np.array(Traj)
    dts = (traj[1:] - traj[:traj.shape[0] - 1])[:, 6]
    controls = traj[: traj.shape[0] -1][:, 7 : 10]
    control_mags = np.linalg.norm(controls, axis=1)
    
    print(np.dot(control_mags, dts))
    print(tf * Umax)


    print(f'Percentage of fuel used: {100 * np.dot(control_mags, dts) / (tf * Umax)}' )
    print(f'Raw fuel used: {np.dot(control_mags, dts)}')


    print(BoundaryFirst)
    print(Traj[0])
    print(Traj[-1])
    print(BoundaryLast)

    Traj_array = np.array(Traj).T

    plot_traj(Traj_array)



    Full_Plot(Traj,ref_state,ref_state)
    '''

    compute_trajectories()