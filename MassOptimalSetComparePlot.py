import scipy.io
import numpy as np
import numpy.linalg as la
from sympy import *
import os
import os.path
from scipy.linalg import lu_factor, lu_solve, eigh
import matplotlib
import matplotlib.pyplot as plt
import itertools
from scipy.interpolate import CubicSpline

######
######
###### COMMENT: THIS CODE IS WORKING BUT REQUIRES DATA FROM FOLDERS TO WORK
######
######

###############################################################################

def Reachability_Plot(ref_state, ref_STM, ref_STT, FileName_mass_data, J_max, dir1, dir2):

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
    ax1.scatter(data[:,0] ,data[:,1] , color=[9/255,83/255,186/255], marker='o')
    # specify legend and axes
    ax1.legend(["Reference Condition","Mass Optimal Conditions"], loc='upper right')
    ax1.set_xlabel(r'$\delta X$ [DU]')
    ax1.set_ylabel(r'$\delta Y$ [DU]')

    # ax2 information
    FileName_test = "./Reachable_Trajectories/TrajI"
    # first plot the reference trajectory
    ax2.plot(ref_state[:,0], ref_state[:,1], color=[0/255,0/255,0/255])
    # Plot the mass-optimal reachable set
    # Loop to plot all of the optimized trajectories
    fail_count = 0
    for l in range(max(np.shape(data))):
    #for l in range(10):
        enumerated_file = f"{FileName_test}_{l}.mat"
        if not os.path.isfile(enumerated_file):
            fail_count += 1
        else:
            TrajI = list(scipy.io.loadmat(enumerated_file).values())[-1]
            ax2.plot(TrajI[:,0], TrajI[:,1], color=[9/255,83/255,186/255], zorder = 2)
    # Plot the energy-optimal reachable set
    ellipse_info = projection(ref_STM,ref_STT,ref_state,J_max,dir1,dir2)
    ellipse_info_interpolated = interpolate_ellipses(ellipse_info,dir1,dir2,ax2)

    ax2.plot(ref_state[:,0], ref_state[:,1], color=[0/255,0/255,0/255], zorder = 4)
    ax2.grid(True)

    ref_patch = matplotlib.lines.Line2D([], [], color=[0/255,0/255,0/255], label='Reference')
    mass_patch = matplotlib.patches.Patch(color=[9/255,83/255,186/255], label='Mass Optimal Set')
    energy_patch = matplotlib.patches.Patch(color=[252/255, 186/255, 3/255], label='Energy Optimal Set')

    leg = ax2.legend(handles=[ref_patch, mass_patch, energy_patch], loc='upper right')
    leg.set_zorder(5)
    ax2.set_xlabel(r'$X$ [DU]')
    ax2.set_ylabel(r'$Y$ [DU]')

    fig.set_size_inches(10.5, 5.5, forward=True)
    fig.set_tight_layout(True)
    #plt.savefig('SetComparison.png', dpi=500)
    #plt.savefig('SetComparison.pdf')
    plt.show()

###############################################################################

def projection(STM_full,STT_full,state_full,J_max,dim1,dim2):
    # This function takes STMs and STTs for a particular initial condition and makes a
    # projection plot in dim1,dim2 space of the hyperellipsoids for each sampled point.

    # INPUTS: 
    # STM_full is the set of state transition matrices starting from t = 0 to t = end time
    # STT_full is the set of state transition tensors starting from t = 0 to t = end time
    # state_full is the set of states starting from t = 0 to t = end time
    # J_max is the max energy cost associated with the orbit
    # dim1 and dim2 are strings describing the plot dimensions on the x-axis and y-axis, respectively
        # dim1 MUST be either 'X', 'Y', 'Z', 'Xdot', 'Ydot', and 'Zdot'. Same goes for dim2. 
    
    # OUTPUT:
    # ellipse_info is an array of size [rows of state_full x 5] containing info for ellipses. Columns (in order) are :
        # dim1-coordinate center (ex: x-coordinate center, xdot coordinate center, zdot coordinate center)
        # dim2-coordinate center
        # width (so 2 times the semi-major axis)
        # height (so 2 times the semi-minor axis)
        # rotation angle (radians) from x-axis to semi-major axis. Bounded between 0 and pi.

    # plt.title('Hyperellipsoid Projections in 2D Plane')
    ellipses = []

    # establish projection matrix and relevant indices for plotting based on dim1 and dim2
    A = np.zeros((2,6)) 
    if dim1 == 'X':
        s1 = 0
    elif dim1 == 'Y':
        s1 = 1
    elif dim1 == 'Z':
        s1 = 2
    elif dim1 == 'Xdot':
        s1 = 3
    elif dim1 == 'Ydot':
        s1 = 4
    elif dim1 == 'Zdot':
        s1 = 5

    if dim2 == 'X':
        s2 = 0
    elif dim2 == 'Y':
        s2 = 1
    elif dim2 == 'Z':
        s2 = 2
    elif dim2 == 'Xdot':
        s2 = 3
    elif dim2 == 'Ydot':
        s2 = 4
    elif dim2 == 'Zdot':
        s2 = 5

    # place 1's in relevant positions of projection matrix A:
    A[0,s1] = 1
    A[1,s2] = 1

    #ref_trajectory, = plt.plot(state_full[:,s1], state_full[:,s2], color=[0,0,0], label='Reference Trajectory') # plot the reference trajectory in black

    [rows,columns,depth] = STM_full.shape # dimensions of STM_full
    ellipse_info = np.zeros((rows,5)) # initialize ellipse_info matrix

    for i in range(0,rows):

        STM = STM_full[i,:,:] @ STM_full[-1,:,:] @ la.inv(STM_full[i,:,:]) # STM for current timestep
        STT = STT_full[i,:,:,:] # STT for current timestep

        Matrix1 = np.block([[np.identity(6), np.zeros((6,6))],
        [-la.solve(STM[:6, 6:12], STM[:6, :6]), la.inv(STM[:6, 6:12])]]) # construct Matrix1 from STMS

        # work with 13 x  13 matrices:
        # TempMatrix1 = la.inv(np.transpose(STM_full[i,:,:])) @ (STT_full[-1,12,:,:] - STT_full[i,12,:,:]) @ la.inv(STM_full[i,:,:])
        # TempMatrix2 = np.transpose(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))@STT_full[i,12,:,:]@(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))+TempMatrix1

        # work with 12 x 12 matrices:
        TempMatrix1 = la.inv(np.transpose(STM_full[i,:12,:12])) @ (STT_full[-1,12,:12,:12] - STT_full[i,12,:12,:12]) @ la.inv(STM_full[i,:12,:12])
        TempMatrix2 = np.transpose(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))@STT_full[i,12,:12,:12]@(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))+TempMatrix1
        Matrix2 = TempMatrix2[:12,:12]

        # eigh returns eigenvalues in ascending order with eigenvectors in their corresponding positions
        E = np.transpose(Matrix1) @ Matrix2 @ Matrix1  # establish E-matrix
        E_star = np.block([[np.identity(6), np.identity(6)]]) @ E @ np.transpose(np.block([[np.identity(6), np.identity(6)]])) # establish Estar matrix
        gamma, w = eigh(E_star) # get eigenvalues and eigenvectors of E_star
        
        # Adjustments to compensate for degeneracy:
        w_adj = w[:,1:6] # remove first eigenvector (the one with infinite extent)
        gamma_adj = gamma[1:6] # remove first eigenvalue (the one with infinite extent)
        Qprime = np.transpose(w_adj) # establish Q matrix with row-vectors corresponding to eigenvectors 
        Eprime = np.diag(gamma_adj) # form adjusted eigenvalue matrix

        lam, z = eigh(Qprime @ np.transpose(A) @ A @ np.transpose(Qprime), Eprime) # solve adjusted problem for new eigenvalues and eigenvectors
        
        # normalize eigenvectors
        z1 = z[:,3] / la.norm(z[:,3]) 
        z2 = z[:,4] / la.norm(z[:,4])

        # get directions of projected energy ellipsoid
        dir1 = A @ (np.transpose(Qprime)) @ z1 
        dir2 = A @ (np.transpose(Qprime)) @ z2
        angle = np.arctan2(dir1[1],dir1[0]) # radians, angle of ellipse rotation


        ext1 = 2*np.sqrt(2*J_max*lam[3]) #  width of ellipse projection
        ext2 = 2*np.sqrt(2*J_max*lam[4]) # height of ellipse projection

        rotation = angle*180/np.pi

        position = np.array([state_full[i,s1],state_full[i,s2]])
        
        # ellipse plotting:
        ellipse = matplotlib.patches.Ellipse(xy=(position[0],position[1]), width=ext1, height=ext2, edgecolor=[252/255, 186/255, 3/255], fc=[252/255, 186/255, 3/255], angle=rotation, zorder=3)

        #ax.add_patch(ellipse)
        ellipses.append(ellipse)
            

        # ellipse_info is n x 5 array storing the x-coordinate of center, y-coordinate of center, semi-major axis, semi-minor axis, and rotation angle
        ellipse_info[i,0] = position[0]
        ellipse_info[i,1] = position[1]

        # Ensure larger extent is included in column 2 of ellipse_info:
        if ext1 > ext2: # extent 1 is larger extent
            ellipse_info[i,2] = ext1
            ellipse_info[i,3] = ext2
            ellipse_info[i,4] = np.arctan2(dir1[1],dir1[0]) # rad, rotation angle from x-axis to semi-major axis
        else: # extent 2 is larger extent
            ellipse_info[i,2] = ext2
            ellipse_info[i,3] = ext1
            ellipse_info[i,4] = np.arctan2(dir2[1],dir2[0]) # rad, rotation angle from x-axis so semi-major axis

        if ellipse_info[i,4] < 0:
            ellipse_info[i,4] = ellipse_info[i,4] + np.pi # correction so that ellipse rotation angle is between 0 and pi
            

    return ellipse_info

###############################################################################

def interpolate_ellipses(ellipse_info,p1,p2,ax):
    # This function takes an array containing information describing ellipses and creates an array of interpolated ellipses using CubicSpline.
    # It will also make a plot containing the reference trajectory and the projections. 
        # Ideally, enough interpolation should be performed so that individual ellipses
        # are indistinguishable from each other in the unzoomed figure that is produced. 
        # It should look like 1 continuous blob around the reference trajectory. 

    # ellipse_info is a rows x 5 numpy array containing ellipse characteristics
        # First column contains p1-coordinates of ellipses' centers
        # Second column contains p2-coordinates of ellipses' centers
        # Third column contains the first extent of the ellipse
        # Second column contains the second extent of the ellipse
        # Fifth column contains the rotation angle (in radians) of the ellipse
    # p1 is a string describing the x-axis of the projection space. MUST BE 'X','Y','Z', 'Xdot','Ydot', or 'Zdot'. (Upper-case matters)
    # p2 is a string describing the y-axis of the projection space. MUST BE 'X','Y','Z', 'Xdot','Ydot', or 'Zdot'. (Upper-case matters)

    # ellipse_info_ip is the output of the function and is an n x 5 array describing the characteristics of the interpolated ellipses
        # Columns of ellipse_info_ip contain characteristics in same order as ellipse_info

    [rows,columns] = ellipse_info.shape # dimensions of ellipse_info
    n = 30*rows # number of samples in interpolation vector
    x = np.linspace(0,rows,num = rows) # vector of sampled points in original array
    xs = np.linspace(0,rows, num = n) # vector of sampled points in interpolated array (len(xs) > len(x))

    h = ellipse_info[:,0] # p1-coordinate centers of ellipses
    k = ellipse_info[:,1] # p2-coordinate centers of ellipses
    ext1 = ellipse_info[:,2] # first extent of ellipses
    ext2 = ellipse_info[:,3] # second extent of ellipses
    r = ellipse_info[:,4] # rotation angle of ellipses

    # Unwrapping-angles procedure (prevents huge drops or rises in angle in r vector):
    r_new = np.zeros((rows)) # create empty vector
    r_new[0] = r[0] # initialize r_new[0]

    for i in range(1,rows):
        difference = r[i] - r_new[i-1] # difference between current angle and previous angle

        if difference > np.pi / 2: # check if difference is very large
            r_new[i] = r[i] - np.pi # unwrap angle
        elif difference < -np.pi / 2: # check if difference is very large (but negative this time)
            r_new[i] = r[i] + np.pi # unwrap angle
        else:
            r_new[i] = r[i] # keep angle if differnence is very small

    r = r_new # reset r to r_new
    # The above procedure is needed to prevent r vector (converted to degrees) from looking like, for example, [...,178,179.5,1,2,...].
    # This would cause problems for interpolation. It would convert above example to [...178,179.5,181,182,...], which is now 
    # interpreted (correctly) as a small angle difference between vector entries.

    # interpolate points using CubicSpline:
    h_ip = CubicSpline(x,h)(xs)
    k_ip = CubicSpline(x,k)(xs)
    ext1_ip = CubicSpline(x,ext1)(xs)
    ext2_ip = CubicSpline(x,ext2)(xs)
    r_ip = CubicSpline(x,r)(xs)

    # fill in information for array of interpated ellipses:
    ellipse_info_ip = np.zeros((n,5))
    ellipse_info_ip[:,0] = h_ip
    ellipse_info_ip[:,1] = k_ip
    ellipse_info_ip[:,2] = ext1_ip
    ellipse_info_ip[:,3] = ext2_ip
    ellipse_info_ip[:,4] = r_ip

    # create plot characteristics
    #fig, ax = plt.subplots()
    #ax = plt.gca()
    # plt.title('Hyperellipsoid Projections in 2D Plane')

    # determine max and min characteristics of ellipses for purposes of establishing axis limits on figure
    hmax = np.max(ellipse_info[:,0])
    hmin = np.min(ellipse_info[:,0])
    kmax = np.max(ellipse_info[:,1])
    kmin = np.min(ellipse_info[:,1])
    amax = np.max(ellipse_info[:,2])

    # set up axis characteristics. (Uncomment if you want to fix axes and/or set axes equal)
    #ax.set_xlim([hmin-1.25*amax, hmax+1.25*amax]) 
    #ax.set_ylim([kmin-1.25*amax, kmax+1.25*amax])
    #ax.set_aspect('equal', adjustable='box') 

    #ref_trajectory, = plt.plot(h, k, color=[0,0,0], label='Reference Trajectory') # reference trajectory points

    # plot interpolated ellispes:
    for i in range(0,n):
        ellipse = matplotlib.patches.Ellipse(xy=(h_ip[i],k_ip[i]), width=ext1_ip[i], height=ext2_ip[i], edgecolor=[252/255, 186/255, 3/255], fc=[252/255, 186/255, 3/255], angle=r_ip[i]*(180/np.pi), zorder=3)
        ax.add_patch(ellipse)

    return ellipse_info_ip

# FUNCTIONS
# Detailed function descriptions are present inside functions. A brief summary is shown below:

# projection 
# Takes in array of STMs, STTs, states. It also takes in energy cost J_max. It takes
# dimension 1 and dimension 2 of projection space which must be 'X','Y','Z','Xdot','Ydot','Zdot'
# It outputs ellipse_info, which is an n x 5 array (where n is the number of STMs).
# See the function for description of columns of the array.

# tangent_lines
# Takes in ellipse_info array (output of projection function). Also specifiy axes labels,
# which must follow same convention as dim1 and dim2 from projection function
# outputs solution1_array and solution2_array, which describes solutions to 2*(n-1) tangent line problems
# where n is the number of STMs again.

# interpolate_ellipses
# Takes in ellipse_info array (output of projection function. Specify axes labels in the same manner
# as previous functions.
# Returns interpolated_ellipses array which has characteristics of ellipses determined through CubicSpline interpolation


# function calls are below (must define STM_full,STT_full,J_max)
#ellipse_info = projection(STM_full,STT_full,state_full,J_max,'X','Y')
#limits = [0,259]
#[solution1_array,solution2_array] = tangent_lines(ellipse_info,limits,"X","Y")
#ellipse_info_interpolated = interpolate_ellipses(ellipse_info,'X','Y')



### LOAD REFERENCE TRAJECTORY STATE AND PERIOD ###
J_max = 3.514110698664422E-04 
# Set file name to save data on first run
FileName_state = "EnergyOptimal_state_EarthMoon_L2nrho.mat"
FileName_STM = "EnergyOptimal_STM_EarthMoon_L2nrho.mat"
FileName_STT = "EnergyOptimal_STT_EarthMoon_L2nrho.mat"
# load data
state_full = list(scipy.io.loadmat(FileName_state).values())[-1]
STM_full = list(scipy.io.loadmat(FileName_STM).values())[-1]
STT_full = list(scipy.io.loadmat(FileName_STT).values())[-1]

FileName_data = "Mass_Reachability_95percent.mat"
Reachability_Plot(state_full, STM_full, STT_full, FileName_data, J_max, 'X', 'Y')
