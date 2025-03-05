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
###### COMMENT: THIS CODE IS MOST ROBUST AND COMPLICATED VERSION TO UNDERSTAND
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
    phase.optimizer.set_PrintLevel(4)
    phase.PrintMeshInfo = False
    flag = phase.optimize()

    return phase, flag

###############################################################################

def run_code(ref_state, tf, ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl):
    ### RUNNING OPTIMIZATION ###
    # Run the energy-optimal version
    try:
        E_or_T = 0
        phase, flag1 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
        Traj = phase.returnTraj()
        IG = Traj
        TrajE = Traj
        # Now run the thrust-optimal version
        E_or_T = 1
        phase2, flag2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
        Traj = phase2.returnTraj()

        ### REINTEGRATION ###
        if(phase2.MeshConverged and flag2==0): # this line is satisfied if the mesh has converged and is optimal
            Tab = phase2.returnTrajTable()
            # Reintegration
            integTab = ode.integrator(0.00001,Tab)
            integTab.setAbsTol(1.0e-16)
            integTab.setRelTol(1.0e-14)
            # Find reintegrated thrust-optimal trajectory (TrajI)
            TrajI   = np.array(integTab.integrate_dense(Traj[0],tf))

            ### FINDING VARS FOR FITNESS ###
            deviation = np.reshape(np.array(BoundaryFirst[:6]),(1,6)) - np.array(ref_state[0,:6])
            # Below is one method to calculate dV
            Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
            # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
            dV = scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1] # in DU/TU

            ### VERIFY ACCURACY ###
            # This loop is meant to verify that the reintegrated trajectory is accurate
            if np.linalg.norm(TrajI[-1][:6] - TrajI[0][:6]) < MeshTol     and     max(Thrust_Mag) < Umax:
                VerifyParam = 1
            else:
                VerifyParam = 0
        else:
            dV = 0
            deviation = BoundaryFirst[:6]
            VerifyParam = 0
            TrajI = 0
    except:
        dV = 0
        deviation = BoundaryFirst[:6]
        VerifyParam = 0
        TrajI = 0
    return dV, deviation, VerifyParam, TrajI, TrajE

###############################################################################

def PSO_Plot(max_iterations,num_particles,Fitness,VerifyParam,X):

    fig = plt.figure()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3d = plt.subplot(122, projection='3d')

    # ax1 will handle the percent convergence across iterations
    # ax2 will show the average fitness across iterations
    conv_percent = np.zeros(max_iterations)
    avg_fitness = np.zeros(max_iterations)        

    for j in range(max_iterations):
        Net_Fit = 0.0
        for m in range(num_particles):
                if VerifyParam[0,m,j] == 1:
                    ax3d.scatter(X[0,m,j], X[1,m,j], X[2,m,j], alpha=(j+1)/max_iterations, color=[0,0,0], marker='^')
                    Net_Fit = Net_Fit + Fitness[0,m,j]
                else:
                    ax3d.scatter(X[0,m,j], X[1,m,j], X[2,m,j], alpha=(j+1)/max_iterations, color=[1,0,0], marker='o')
        conv_percent[j] = 100*np.sum(VerifyParam[:,:,j])/num_particles
        avg_fitness[j] = Net_Fit/np.sum(VerifyParam[0,:,j])
    
    black_tri = ax3d.scatter([], [], alpha=1, color=[0,0,0], marker='^', label='Converged')
    red_circ = ax3d.scatter([], [], alpha=1, color=[1,0,0], marker='o', label='Unconverged')
    ax3d.legend(handles=[black_tri, red_circ])

    ax3d.set_xlabel('X [DU]')
    ax3d.set_ylabel('Y [DU]')
    ax3d.set_zlabel('Z [DU]')

    ax1.plot(conv_percent, color=[130/255,130/255,130/255])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Percent Converged')

    ax2.plot(avg_fitness, color=[130/255,130/255,130/255])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Fitness')

    fig.set_size_inches(10.5, 5.5, forward=True)
    fig.set_tight_layout(True)
    #plt.savefig('PSO_Example.pdf')
    plt.show()

###############################################################################

def Full_Plot(Traj,IG,ref_state):

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    Traj_array = np.array(Traj).T
    IG_array = np.array(IG).T

    Thrust_Energy_Mag = DU/TU**2 * np.array([[np.sqrt(IG_array[7,i]**2 + IG_array[8,i]**2 + IG_array[9,i]**2)] for i in range(max(np.shape(IG_array)))])
    Thrust_Mass_Mag = DU/TU**2 * np.array([[np.sqrt(Traj_array[7,i]**2 + Traj_array[8,i]**2 + Traj_array[9,i]**2)] for i in range(max(np.shape(Traj_array)))])

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122, projection='3d')

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7], color=[9/255,83/255,186/255])  # plots u_x vs time
    ax0.plot(TU/86400 * IG_array[6], DU/TU**2 * IG_array[7], color='r', color=[252/255, 186/255, 3/255], linestyle='dashed')
    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8], color=[9/255,83/255,186/255])  # plots u_y vs time
    ax1.plot(TU/86400 * IG_array[6], DU/TU**2 * IG_array[8], color='r', color=[252/255, 186/255, 3/255], linestyle='dashed')
    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9], color=[9/255,83/255,186/255])  # plots u_z vs time
    ax2.plot(TU/86400 * IG_array[6], DU/TU**2 * IG_array[9], color='r', color=[252/255, 186/255, 3/255], linestyle='dashed')
    ax3.plot(TU/86400 * Traj_array[6], Thrust_Mass_Mag, color=[9/255,83/255,186/255])     # plots u mag vs time
    ax3.plot(TU/86400 * IG_array[6], Thrust_Energy_Mag, color='r', color=[252/255, 186/255, 3/255], linestyle='dashed')     # plots u mag vs time

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7], color=[9/255,83/255,186/255])  # plots u_x vs time
    ax0.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[7], color=[252/255, 186/255, 3/255], linestyle='dashed')
    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8], color=[9/255,83/255,186/255])  # plots u_y vs time
    ax1.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[8], color=[252/255, 186/255, 3/255], linestyle='dashed')
    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9], color=[9/255,83/255,186/255])  # plots u_z vs time
    ax2.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[9], color=[252/255, 186/255, 3/255], linestyle='dashed')
    ax3.plot(TU/86400 * Traj_array[6], Thrust_Mass_Mag, color=[9/255,83/255,186/255])     # plots u mag vs time
    ax3.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), Thrust_Energy_Mag, color=[252/255, 186/255, 3/255], linestyle='dashed')     # plots u mag vs time


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
    plt.show()

###############################################################################

def Reachability_Plot(ref_state,tf,ode,IG,FileName_mass_data):

    data = list(scipy.io.loadmat(FileName_mass_data).values())[-1]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.rcParams['text.usetex'] = True

    # turn on grid
    ax1.grid(True)
    # plot the reference trajectory
    ax1.scatter(0.0, 0.0, color=[0/255,0/255,0/255])
    # plot the reachable initial conditions for mass optimality
    ax1.scatter(data[:,0] ,data[:,1] , color=[130/255,130/255,130/255], marker='o')
    # specify legend and axes
    ax1.legend(["Reference Condition","Mass Optimal Conditions"], loc='upper right')
    ax1.set_xlabel(r'$\delta X$ [DU]')
    ax1.set_ylabel(r'$\delta Y$ [DU]')

    # Set parameters for optimization
    optType = "LGL3"
    Umax = 0.0184  # DU/TU^2
    Umin = 1.0e-8
    numKnots = 64
    numThreads = 8
    MeshTol = 1.0e-10
    EControl = 1.0e-12

    FileName_test = "./Reachable_Trajectories/TrajI"

    ax2.plot(ref_state[:,0], ref_state[:,1], color=[0/255,0/255,0/255])
    # Loop to plot all of the optimized trajectories; runtime is ~ 1.5 seconds per trajectory
    fail_count = 0
    for l in range(max(np.shape(data))):
    #for l in range(10):
        enumerated_file = f"{FileName_test}_{l}.mat"
        if not os.path.isfile(enumerated_file):
            BoundaryFirst = list(data[l][:6] + IG[0][:6]) + [0]
            BoundaryLast = BoundaryFirst[:6] + [tf]
            dV, deviation, VerifyParam, TrajI, TrajE = run_code(ref_state, tf, ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
            if VerifyParam == 1:
                print('Plotting ',l,'/',max(np.shape(data)))
                scipy.io.savemat(enumerated_file, {"TrajI": TrajI})
                ax2.plot(TrajI[:,0], TrajI[:,1], color=[130/255,130/255,130/255])
            else:
                fail_count += 1
        else:
            TrajI = list(scipy.io.loadmat(enumerated_file).values())[-1]
            ax2.plot(TrajI[:,0], TrajI[:,1], color=[130/255,130/255,130/255])

    ax2.plot(ref_state[:,0], ref_state[:,1], color=[0/255,0/255,0/255])
    ax2.grid(True)

    ax2.legend(["Reference", "Mass Optimal Trajectories"], loc='upper right')
    ax2.set_xlabel(r'$X$ [DU]')
    ax2.set_ylabel(r'$Y$ [DU]')


    fig.set_size_inches(10.5, 5.5, forward=True)
    fig.set_tight_layout(True)
    plt.savefig('MassOptimalReachability.png', dpi=500)
    plt.savefig('MassOptimalReachability.pdf')
    plt.show()



###############################################################################

def Compare_Plot(Traj,IG,ref_state):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # plot the reference trajectory
    ax.plot(ref_state[:,0], ref_state[:,1], ref_state[:,2], color=[0,0,0])

    # plot the energy optimal trajectory
    IG_array = np.array(IG)
    ax.plot(IG_array[:,0], IG_array[:,1], IG_array[:,2], color=[109/255,0,234/255])

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

    ### SET PARAMETERS ###
    # These parameters will remain constant for every iteration and do not need updating
    optType = "LGL3"
    Umax = 0.0184           # DU/TU^2
    Umin = 1.0e-8
    numKnots = 64
    numThreads = 8
    MeshTol = 1.0e-08
    EControl = 1.0e-10
    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)
    mu_star =  0.01215059   # Constant for CR3BP

    ### LOAD REFERENCE TRAJECTORY STATE AND PERIOD ###
    tf  = 2.085034838884136 # TU
    # Load in the reference state data
    FileName_ref = "C:/Users/ccm41/Downloads/Research/Starlift/STM_Stuff/STMPrecompute-main/STMPrecompute-main/EnergyOptimal_state_EarthMoon_L2nrho.mat"
    ref_state = list(scipy.io.loadmat(FileName_ref).values())[-1]

    ### LOAD BASE INITIAL GUESS ###
    # IG takes the form of [state,time,control]
    IG = [[ref_state[i,0], ref_state[i,1], ref_state[i,2], ref_state[i,3], ref_state[i,4], ref_state[i,5], tf*i/(max(np.shape(ref_state))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state)))]
    ode = CR3BP_Thrust_Dynamics(mu_star)

    ### LOAD DATA FOR OUTPUT ###
    FileName_data = "C:/Users/ccm41/Downloads/Research/Starlift/ASSET_Stuff/asset_asrl/Mass_Reachability_95percent.mat"
    data = list(scipy.io.loadmat(FileName_data).values())[-1]
    #scipy.io.savemat(FileName_data, {"initial_states": data})

    # Inputs to the overhead while loop
    angle = 202 #deg
    weight = np.array([np.cos(angle*np.pi/180), np.sin(angle*np.pi/180), 0, 0, 0, 0])
    #optimal_state = np.array([0.01084709,  0.00282025,  0.01001525,  0.00266345, -0.01583759, 0.0013435]) # J/Jmax = 0.999; maximum for angle = 0.0
    #optimal_state = np.array([0.00412008,  0.02031663,  0.00239706,  0.02149218, -0.00249428, -0.01112556]) # J/Jmax = 0.985; maximum for angle = 5.0
    #optimal_state = np.array([0.01067089,  0.00241113,  0.00975311,  0.00399922, -0.01416087, -0.00640567]) #J/Jmax = 0.961; guess for angle = 10.0
    #optimal_state = np.array([0.01070352,  0.0024651 ,  0.00978868,  0.00390386, -0.01448702, -0.00640664]) #J/Jmax = 0.960; guess for angle = 30.0
    #optimal_state = np.array([-0.00038276,  0.01937584, -0.00203271,  0.02172771,  0.0081204, -0.01092992]) #J/Jmax = 0.991; guess for angle = 45
    #optimal_state = np.array([0.00920747,  0.00237285,  0.00779071,  0.0011143 , -0.00760444, -0.00404732]) #J/Jmax = 0.916; guess for angle = 45
    #optimal_state = np.array([0.00914069,  0.00784247,  0.00684051,  0.0083169 , -0.00799557, -0.00675053]) #J/Jmax = 0.917; maximum for angle = 45
    #optimal_state = np.array([0.01075519,  0.00264993,  0.00988391,  0.0036404 , -0.01504075, -0.00632218]) #J/Jmax = 0.960; guess for angle = 60.0
    #optimal_state = np.array([-0.00277333,  0.01888774, -0.00156929,  0.02097652,  0.0114266, -0.00959365]) #J/Jmax = 0.995; guess for angle = 75
    #optimal_state = np.array([0.00227569,  0.01961604,  0.00185603,  0.01981015,  0.00120812, -0.00915852]) #J/Jmax = 0.986; guess for angle = 80
    #optimal_state = np.array([-0.00624806,  0.01803216, -0.00514324,  0.01576087,  0.0070392, -0.00876373]) #J/Jmax = 0.975; guess for angle = 89.9
    #optimal_state = np.array([-0.00838196,  0.01810611, -0.00561519,  0.01869592,  0.01159955, -0.00838786]) #J/Jmax = 0.990; guess for angle = 135
    #optimal_state = np.array([-0.00904553,  0.01738687, -0.0053329 ,  0.01864202,  0.01368102, -0.00818293]) #J/Jmax = 0.995; maximum for angle = 135
    #optimal_state = np.array([-0.01179883, -0.00298069, -0.00997879, -0.00199457,  0.01680163, -0.00275168]) # J/Jmax = 0.997; maximum for angle = 180.0
    #optimal_state = np.array([-0.01180823, -0.00313, -0.01021419, -0.00263853,  0.01665529, -0.0025315]) # J/Jmax = 0.999; maximum for angle = 200.0
    #optimal_state = np.array([-0.00537032, -0.02551035, -0.00107045, -0.02946507,  0.00822487, 0.01658255]) # J/Jmax = 0.995; maximum for angle = 200.0
    #optimal_state = np.array([-0.00342547478, -0.0255382117, -0.0000895040134, -0.0269956496, 0.00599861774,  0.0130596669]) # J/Jmax = 0.996; maxmimum for angle = 210.0
    #optimal_state = np.array([-0.00303216, -0.02496717,  0.00027755, -0.02620111,  0.00438122, 0.01287917]) # J/Jmax = 0.995; guess for angle = 230.0
    #optimal_state = np.array([0.00073709, -0.02461603,  0.00119914, -0.02637356, -0.00162676, 0.01343743]) # J/Jmax = 0.991; maximum for angle = 250.0
    #optimal_state = np.array([0.00796167, -0.00957139,  0.00879103, -0.00824944, -0.01303803, 0.00118188]) # J/Jmax = 0.885; guess for angle = 275.0
    #optimal_state = np.array([0.00073709, -0.02461603,  0.00119914, -0.02637356, -0.00162676, 0.01343743]) # J/Jmax = 0.960; guess for angle = 270.0
    #optimal_state = np.array([0.00828687, -0.0066427 ,  0.00732374, -0.00215118, -0.01206712, -0.00514633]) # J/Jmax = 0.964; guess for angle = 280.0
    #optimal_state = np.array([0.00903995, -0.00540297,  0.00745034, -0.00254331, -0.00972988, -0.00333354]) # J/Jmax = 0.981; maximum for angle = 290.0
    #optimal_state = np.array([0.00761232, -0.0040261 ,  0.00644397, -0.001049  , -0.00407729, -0.00064532]) # J/Jmax = 0.966; maximum for angle = 305.0
    #optimal_state = np.array([0.00761232, -0.0040261 ,  0.00644397, -0.001049  , -0.00407729, -0.00064532]) # J/Jmax = 0.967; maximum for angle = 315.0
    #optimal_state = np.array([0.00713989, -0.00366238,  0.00677496, -0.00199472, -0.00280237, -0.00206927]) # J/Jmax = 0.984; maximum for angle = 320.0 
    
    #optimal_state = np.array([0.00098605,  0.00543685, -0.0028203, -0.00072117, 0.00137802, -0.00109394]) #J/Jmax = 0.95ish? (must check);
    #optimal_state = np.zeros(6)
    #optimal_state = 0.35*(np.array([0.00227569,  0.01961604,  0.00185603,  0.01981015,  0.00120812, -0.00915852]) + np.array([0.01067089,  0.00241113,  0.00975311,  0.00399922, -0.01416087, -0.00640567]))

    #optimal_state = 0.37*(np.array([-0.00904553,  0.01738687, -0.0053329 ,  0.01864202,  0.01368102, -0.00818293]) + np.array([-0.01179883, -0.00298069, -0.00997879, -0.00199457,  0.01680163, -0.00275168]))
    #optimal_state = 0.37*(np.array([-0.009254783064645 ,  0.009393216564916 , -0.006382215951001 ,  0.008570222383114 ,  0.018654509588774,  -0.003777324928620]) + np.array([-0.009235669590537,   0.016446809433260,  -0.005309763576215,   0.017859340867374,   0.013622595092753,  -0.008440605678661]))
    #optimal_state = np.array([-0.00954614,  -0.01093506, 0.00496872, -0.0118053 ,  -0.01176956, 0.00537693])
    #optimal_state = 0.37*(np.array([-0.00937339, -0.01230004, -0.00625955, -0.01076034,  0.01278343, 0.00207435]) + np.array([-0.003027186121811 , -0.025560503353468 ,  0.000020215054497 , -0.026899611724809  , 0.005470563936106  , 0.012927963717551]))

    optimal_state = 0.4*(np.array([-0.009006837598360 , -0.012277453103609,  -0.005647572358757  ,-0.011108379060406 ,  0.012704345287543  , 0.001964015240868]) + np.array([-0.01179883, -0.00298069, -0.00997879, -0.00199457,  0.01680163, -0.00275168]))
    #optimal_state = 0.4*(np.array([0.00073709, -0.02461603,  0.00119914, -0.02637356, -0.00162676, 0.01343743]) + np.array([0.00591227, -0.01655975,  0.00440179, -0.01782453, -0.0042587,  0.0089859]))
    optimal_Fit = 0
    optimal_dV = 0
    counter = 0
    converge_counter = 0

    while optimal_dV/(Umax*tf) < 0.95    and    counter < 3     and     converge_counter < 2:

        ### INITIALIZATION ###
        # runtime per PSO run (minutes) = num_particles * max_iterations / 20
        num_particles = 25
        max_iterations = 15
        beta = 0.7
        delta = 0.5
        alpha = 1
        iter = 0

        # states of particles (delta rv, particle ID, iteration of particle)
        X = np.zeros((6,num_particles,max_iterations))
        dV = np.zeros((1,num_particles,max_iterations))
        deviation = np.zeros((6,num_particles,max_iterations))
        VerifyParam = np.zeros((1,num_particles,max_iterations))
        Fitness = np.zeros((1,num_particles,max_iterations))
        Fitness_Over_Iter = np.zeros((1,max_iterations))

        # For reasonable randomized initial sampling, use following two lines
        for k in range(num_particles):
            # Full random:
            #X[:,k,iter] = np.random.normal(0,3e-3,(6,1))[:6,0]
            # Random around previously found point:
            X[:,k,iter] = optimal_state + np.random.normal(0,8e-4,(6,1))[:6,0]
        # Introduce some best states
        X[:,4,iter] = optimal_state

        while iter < max_iterations and optimal_dV/(Umax*tf) < 0.99:
            for k in range(num_particles):
                ### RUN OPTIMIZER ###
                BoundaryFirst = list(X[:,k,iter] + IG[0][:6]) + [0]
                BoundaryLast = BoundaryFirst[:6] + [tf]
                dV[:,k,iter], deviation[:,k,iter], VerifyParam[:,k,iter], TrajI, TrajE = run_code(ref_state, tf, ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
            
                ### FITNESS CALC ###
                if VerifyParam[:,k,iter] == 1:
                    #Fitness[:,k,iter] = dV[:,k,iter]/(Umax*tf) # Fitness based on dV
                    Fitness[:,k,iter] = np.sum(weight*deviation[:,k,iter]) # Fitness based on deviation
                    # Check to update parameters
                    if Fitness[:,k,iter] > optimal_Fit:
                        optimal_state = deviation[:,k,iter]
                        optimal_Fit = Fitness[:,k,iter]
                        optimal_Traj = TrajI
                        optimal_IG = TrajE
                        optimal_dV = dV[:,k,iter]
                        converge_counter = 0
                        print('Best Fit = ', optimal_Fit, ' ; optimal dV = ',dV[:,k,iter]/(Umax*tf))
                    if dV[:,k,iter]/(Umax*tf) > 0.95:
                        data = np.vstack((data,(deviation[:,k,iter]))) #adds deviation to data

                ### VARIABLE UPDATES ###
                if iter+1 < max_iterations:
                    if alpha > 0.01:
                        alpha = delta**iter
                    X[:,k,iter+1] = (1-beta)*X[:,k,iter] + beta*optimal_state + alpha*np.array([np.random.normal(0,8e-3), np.random.normal(0,8e-4), np.random.normal(0,6e-3), np.random.normal(0,8e-3), np.random.normal(0,1e-2), np.random.normal(0,4e-3)])
                
                print('k = ',k,'; iter = ',iter,'; Fitness = ',Fitness[:,k,iter])

            Fitness_Over_Iter[0,iter] = Fitness[:,:,iter].max()
            iter += 1
        
        scipy.io.savemat(FileName_data, {"initial_states": data}) # save the data (takes ~ 1 millisecond)
        converge_counter += 1
        counter += 1
        print('Finished PSO run number ', counter, ' ; Best Fit = ', optimal_Fit, ' ; optimal dV = ',optimal_dV/(Umax*tf))

    #PSO_Plot(max_iterations,num_particles,Fitness,VerifyParam,X)
    #Full_Plot(optimal_Traj,optimal_IG,ref_state)


    ###########################################################################
    