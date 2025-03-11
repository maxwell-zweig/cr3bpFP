import scipy.io
import numpy as np
import numpy.linalg as la
from sympy import *
import os
import os.path
from scipy.linalg import lu_factor, lu_solve, eigh
from STMint import STMint
import matplotlib
import matplotlib.pyplot as plt
import pdb
import dill #use dill to save lambda-fied class
        
#symbolically define the dynamics for energy optimal control in the cr3bp
#this will be used by the STMInt package for numerically integrating the STM and STT
def optControlDynamics():
    x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En=symbols("x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En")
    #mu = 3.00348e-6
    mu = 0.012150590000000
    mu1 = 1. - mu
    mu2 = mu
    r1 = sqrt((x + mu2)**2 + (y**2) + (z**2))
    r2 = sqrt((x - mu1)**2 + (y**2) + (z**2))
    U = (-1/2)*((x**2) + (y**2)) - (mu1/r1) - (mu2/r2)
    dUdx = diff(U,x)
    dUdy = diff(U,y)
    dUdz = diff(U,z)

    RHS = Matrix([vx,vy,vz,((-1*dUdx) + 2*vy),((-1*dUdy)- 2*vx),(-1*dUdz)])

    variables = Matrix([x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En])

    dynamics = Matrix(BlockMatrix([[RHS - Matrix([0,0,0,lvx,lvy,lvz])], 
        [-1.*RHS.jacobian(Matrix([x,y,z,vx,vy,vz]).transpose())*Matrix([lx,ly,lz,lvx,lvy,lvz])],
        [Matrix([lvx**2+lvy**2+lvz**2])/2]]))
    #return Matrix([x,y,z,vx,vy,vz]), RHS
    return variables, dynamics 




# This function iterates on the initial conditions of the state to reduce the reference orbit error
def state_iterate(target_state, tolerance, stepsize, T_final):
    # load in threeBodyInt class if it doesn't exist
    FileName_threeBodyInt = "./EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle"
    if not os.path.isfile(FileName_threeBodyInt):
        variables, dynamics = optControlDynamics()
        threeBodyInt = STMint(variables, dynamics, variational_order=1) #this is the expensive line

        with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'wb') as threeBodyInt_file:
            dill.dump(threeBodyInt, threeBodyInt_file)

    with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'rb') as threeBodyInt_file:
        threeBodyInt = dill.load(threeBodyInt_file)

    state = target_state
    costates = np.array([[0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000]])
    i = 0
    residual = target_state
    STM = np.random.rand(13,13)

    # begin Newton-Raphson iteration
    while la.norm(la.inv(STM[:6, :6] - np.identity(6)) @ residual) > tolerance and i < 20:
        [state_output, STM] = threeBodyInt.dynVar_int([0,T_final], np.reshape(np.vstack((state,costates,np.array([0]))), (1,13)), output='final', max_step=stepsize, rtol=1e-12, atol=1e-12)
        residual = np.array(state_output[:6],ndmin=2).T - state
        state -= la.inv(STM[:6, :6] - np.identity(6)) @ residual 
        i += 1
        print(str(i))
        print(str(la.norm(la.inv(STM[:6, :6] - np.identity(6)) @ residual)) )

    return state_output




# This function computes the cost of the orbit and returns the costates for continuation purposes
def true_cost(target_state, u_inherent, tolerance, stepsize, T_final, costate_guess):
    # load in threeBodyInt class if it doesn't exist
    FileName_threeBodyInt = "./EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle"
    if not os.path.isfile(FileName_threeBodyInt):
        variables, dynamics = optControlDynamics()
        threeBodyInt = STMint(variables, dynamics, variational_order=1) #this is the expensive line

        with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'wb') as threeBodyInt_file:
            dill.dump(threeBodyInt, threeBodyInt_file)

    with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'rb') as threeBodyInt_file:
        threeBodyInt = dill.load(threeBodyInt_file)

    # set initial guesses for iteration
    costates_iterate = costate_guess
    i = 0
    residual = target_state
    STM = np.random.rand(13,13)

    # begin Newton-Raphson iteration
    while la.norm(la.inv(STM[:6, 6:12]) @ residual) > tolerance and i < 20:
        [state, STM] = threeBodyInt.dynVar_int([0,T_final], np.reshape(np.vstack((target_state,costates_iterate,np.array([0]))), (1,13)), output='final', max_step=stepsize)
        residual = np.array(state[:6],ndmin=2).T - target_state
        costates_iterate -= la.inv(STM[:6, 6:12]) @ residual 
        i += 1
        print(str(i))
        print(str(la.norm(la.inv(STM[:6, 6:12]) @ residual)) )

    # find the true cost
    J_computed = state[12]

    return J_computed, costates_iterate










##############################
###### LINEAR ANALYSIS #######
##############################

#For Earth-Moon, these are the conversions:
# 1 TU = 2.360584684800000E+06/(2*np.pi) seconds
# 1 DU = 384400000.0000000 meters

# Set initial conditions 
mu = 0.012150590000000
ics = [1.06315768e+00,  3.26952322e-04, -2.00259761e-01, 3.61619362e-04, -1.76727245e-01, -7.39327422e-04, 0, 0, 0, 0, 0, 0, 0] #(use this for _L2nrho)
T_final = 2.085034838884136
exponent = 8
t_step = T_final/2.**exponent


# Set file name to save data on first run
FileName_state = "./EnergyOptimal_state_EarthMoon_L2nrho.mat"
FileName_STM = "./EnergyOptimal_STM_EarthMoon_L2nrho.mat"
FileName_STT = "./EnergyOptimal_STT_EarthMoon_L2nrho.mat"

# Run if the file does not exist
if not os.path.isfile(FileName_state):
    variables, dynamics = optControlDynamics()
    threeBodyInt = STMint(variables, dynamics, variational_order=2)

    [state, STM, STT, time] = threeBodyInt.dynVar_int2([0,T_final], ics, output='all', max_step=t_step)

    scipy.io.savemat(FileName_state, {"state": state})
    scipy.io.savemat(FileName_STM, {"STM": STM})
    scipy.io.savemat(FileName_STT, {"STT": STT})
    

# load data
state_full = list(scipy.io.loadmat(FileName_state).values())[-1]
STM_full = list(scipy.io.loadmat(FileName_STM).values())[-1]
STT_full = list(scipy.io.loadmat(FileName_STT).values())[-1]

state = state_full[-1,:]
STM = STM_full[-1,:,:]
STT = STT_full[-1,:,:,:]
time = []
for i in range(len(state_full)):
    time.append(i/len(state_full))


# Compute the E matrix 
Matrix1 = np.block([[np.identity(6), np.zeros((6,6))],
    [-la.solve(STM[:6, 6:12], STM[:6, :6]), la.inv(STM[:6, 6:12])]])
Matrix2 = STT[12, :12, :12]
E = np.transpose(Matrix1) @ Matrix2 @ Matrix1 

E_star = np.block([[np.identity(6), np.identity(6)]]) @ E @ np.transpose(np.block([[np.identity(6), np.identity(6)]]))

# Determine eigenvalues of E matrix
# gamma is the list of eigenvalues in ascending order
# The normalized eigenvector corresponding to the eigenvalue gamma[i] is the column w[:,i]
gamma, w = eigh(E_star)

if w[0,5] > 0:
    w[:,5] = - w[:,5]

# To check eigenstuff, uncomment these lines
#print("Eigenvalue-Eigenvector pairs of E* are:")
#for i in range(6):
#    print(str(gamma[i]) + ", " + str(w[:,i]) + ", " + str(np.linalg.norm(w[:,i])))

J_max = 3.514110698664422E-04 # set any reasonable value for J_max but make sure it can be validated
for i in range(6):
    print("The extent is " + str(la.norm(np.sqrt(2*J_max/gamma[i])*w[:,i])) + " and the direction is " + str(w[:,i]))














################################
###### OBTAIN STATE DATA #######
################################
# Improvement: Make this a function that can just be called rather than commenting in/out
"""
FileName_data = "./state_dataset.mat"

x_dataset = np.zeros((6,260,100000))
# Run if the file does not exist
if not os.path.isfile(FileName_data):
    for i in range(100000):
        rand_vec = np.reshape(np.hstack((np.array(w[:,1]), np.array(w[:,2]), np.array(w[:,3]), np.array(w[:,4]), np.array(w[:,5]))), (6,5)) @ np.random.standard_normal(5)
        rand_vec = np.reshape(rand_vec,(6,1))
        rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
        scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
        dx_0 = rand_vec * scaling_factor
        costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
        x = np.zeros((6,260))
        for index in range(len(state_full)):
            dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
            x[:,index] = state_full[index,:6] + np.reshape(dx, (6,))
        x_dataset[:,:,i] = x
        if i/1000 == 1 or i/10000 ==1 or i/20000 == 1:
            print(i)
    scipy.io.savemat(FileName_data, {"State_Data": x_dataset})

# load data
x_dataset = list(scipy.io.loadmat(FileName_data).values())[-1]
"""









"""
##################################
###### SAMPLED YELLOWPLOTS #######
##################################
# Improvement: Make this call a state_dataset function

# load data
FileName_data = "./state_dataset.mat"
x_dataset = list(scipy.io.loadmat(FileName_data).values())[-1]

fig = plt.figure()
# for 3d plot
ax = fig.add_subplot(projection="3d")
plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
#plt.plot(state_full[:,1], state_full[:,2], color=[0,0,0])
for i in range(1000):
    x = x_dataset[:,:,i]
    #color_random = list(np.random.choice(range(256), size=3)/256)
    color_random = [252/255, 227/255, 3/255]

    # For a 3d orbit plot
    plt.plot(x[0,:], x[1,:], x[2,:], color=color_random)

    # For a 2d orbit plot
    #plt.plot(x[1,:], x[2,:],color=color_random, alpha=1)

# for 2d orbit
#plt.plot(state_full[:,1], state_full[:,2], color=[0,0,0])
plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
#plt.xlabel("Y [DU]")
#plt.ylabel("Z [DU]")
# for 3d orbit
plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
ax.set(
    xlabel='X [DU]',
    ylabel='Y [DU]',
    zlabel='Z [DU]',
)
#plt.savefig('3d_trajectories.png', dpi=500)
plt.show()
"""

















##################################
###### ELLIPSE YELLOWPLOTS #######
##################################
# Improvement: make this work based on method that Jackson wrote up
""" # Does not work currently - VERY close but not quite there
plt.figure()
#plt.xlim(0.985, 1.075)
#plt.ylim(-0.1, 0.1)
ax = plt.gca()
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])

#for index in range(len(state_full)):
for index in range(1):
    A = STM_full[index, (0, 1), :6] + STM_full[index, (0, 1), 6:12] @ la.inv(STM[:6, 6:12]) @ (np.identity(6) - STM[:6, :6])

    # gamma is the list of eigenvalues in ascending order
    # The normalized eigenvector corresponding to the eigenvalue gamma[i] is the column w[:,i]
    gamma_A, w_A = eigh(A.T @ A, E_star)
    semiaxes = np.zeros((2,2))
    k=0

    for i in range(6):
        w_A[:,i] = w_A[:,i]/la.norm(w_A[:,i]) # normalize the eigenvectors

        
        if gamma_A[i] > 1e-10 and gamma_A[i] < 1:

            # This line will scale alpha so it is a semi-axis
            scaling_factor = np.sqrt(2*J_max/(np.transpose(w_A[:,i]) @ E_star @ w_A[:,i]))
            alpha = scaling_factor * A @ w_A[:,i]
            semiaxes[k,:] = alpha
            k+=1

    rotation = 180/np.pi*np.arctan2(semiaxes[0,1], semiaxes[0,0]) # in degrees

    ellipse = matplotlib.patches.Ellipse(xy=(state_full[index,0], state_full[index,1]), width=la.norm(semiaxes[0,:]), height=la.norm(semiaxes[1,:]), edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=rotation)
    ax.add_patch(ellipse)


#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")
#plt.savefig('Ellipse_XY.pdf', format='pdf')
plt.show()
"""



















#################################
###### FILLED YELLOWPLOTS #######
#################################
# Improvement: remove completely after ellipse code because this is expensive and incomplete
"""
fig = plt.figure()
# for 3d plot
#ax = fig.add_subplot(projection="3d")
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
dx_min = np.zeros((6,260))
dx_max = np.zeros((6,260))

#
for i in range(1000):
    rand_vec = np.reshape(np.hstack((np.array(w[:,1]), np.array(w[:,2]), np.array(w[:,3]), np.array(w[:,4]), np.array(w[:,5]))), (6,5)) @ np.random.standard_normal(5)
    rand_vec = np.reshape(rand_vec,(6,1))
    rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
    scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
    dx_0 = rand_vec * scaling_factor
    costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
    x = np.zeros((6,260))
    dx_norm = []
    dlambda_norm = []
    for index in range(len(state_full)):
        dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
        dlambda = STM_full[index,6:12,:6] @ dx_0 + STM_full[index,6:12,6:12] @ costates_initial
        dx_norm.append(la.norm(dx[:3]))
        dlambda_norm.append(la.norm([dlambda[3:]]))
        x[:,index] = state_full[index,:6] + np.reshape(dx, (6,))

    #x_dataset[:,:,i] = x
    dx_matrix = x[:,:] - np.reshape(state_full[:,:6], (6,260))
    for i in range(6):
        for j in range(260):
            if dx_matrix[i,j] < dx_min[i,j]:
                dx_min[i,j] = dx_matrix[i,j]
            if dx_matrix[i,j] > dx_max[i,j]:
                dx_max[i,j] = dx_matrix[i,j]

    #breakpoint()
    #x = np.reshape(x ,(index,6))
    #color_random = list(np.random.choice(range(256), size=3)/256)
    color_random = [252/255, 227/255, 3/255]

    # For a 3d orbit plot
    #plt.plot(x[0,:], x[1,:], x[2,:], color=color_random, alpha=0.2)

    # For a 2d orbit plot
    plt.plot(x[0,:], x[1,:],color=color_random, alpha=1)

    # For a displacement plot
    #plt.plot(time, dlambda_norm,color=color_random)

x_min = np.reshape(state_full[:,:6], (6,260)) + dx_min
x_max = np.reshape(state_full[:,:6], (6,260)) + dx_max

#breakpoint()
#plt.fill_between(state_full[:,0], x_min[1,:], x_max[1,:])
plt.plot(dx_min[0,:], dx_min[1,:])


#for displacement plot
#plt.hlines(0, 0, 1, color = [0,0,0])
#plt.xlabel("Time [TU]")
#plt.ylabel("Thrust Magnitude [DU/TU^2]")
# for 2d orbit
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")
# for 3d orbit
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
#ax.set(
#    xlabel='X [DU]',
#    ylabel='Y [DU]',
#    zlabel='Z [DU]',
#)
#plt.savefig('YellowBounds_XY.pdf', format='pdf')
plt.show()
"""










########################
###### EIGENPLOT #######
########################
# Improvement: Just clean up a bit

TU = 2.360584684800000E+06/(2*np.pi) #seconds
DU = 384400000.0000000 #meters

fig = plt.figure()
plt.rcParams['text.usetex'] = True
plt.hlines(0, 0, T_final*TU/86400, color = [0,0,0])
color_counter = np.zeros((1,6))
for i in [1, 2, 3, 4, 5]:
    rand_vec = w[:,i]
    rand_vec = np.reshape(rand_vec,(6,1))
    rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
    scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
    dx_0 = rand_vec * scaling_factor
    costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
    x = np.zeros((6,260))
    dx_norm = []
    dlambda_norm = []
    for index in range(len(state_full)):
        dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
        dlambda = STM_full[index,6:12,:6] @ dx_0 + STM_full[index,6:12,6:12] @ costates_initial
        # CHANGE depending on what you're plotting
        dx_norm.append(dx[5])
        dlambda_norm.append(la.norm([dlambda[3:]]))
        x[:,index] = state_full[index,:6] + np.reshape(dx, (6,))


    angles = [np.abs(np.dot(rand_vec[:,0],w[:,0])), np.abs(np.dot(rand_vec[:,0],w[:,1])), 
    np.abs(np.dot(rand_vec[:,0],w[:,2])), np.abs(np.dot(rand_vec[:,0],w[:,3])), 
    np.abs(np.dot(rand_vec[:,0],w[:,4])), np.abs(np.dot(rand_vec[:,0],w[:,5]))]

    #breakpoint()
    #x = np.reshape(x ,(index,6))
    #color_random = list(np.random.choice(range(256), size=3)/256)
    #color_random = [252/255, 227/255, 3/255]
    index_max = np.argmax(angles)
    if index_max == 1:
        color_random = '#FFC0CB'
        color_counter = color_counter + np.array([0, 1, 0, 0, 0, 0])
    elif index_max == 2:
        color_random = '#994F00'
        color_counter = color_counter + np.array([0, 0, 1, 0, 0, 0])
    elif index_max == 3:
        color_random = '#006CD1'
        color_counter = color_counter + np.array([0, 0, 0, 1, 0, 0])
    elif index_max == 4:
        color_random = '#E1BE6A'
        color_counter = color_counter + np.array([0, 0, 0, 0, 1, 0])
    elif index_max == 5:
        color_random = '#FF5F1F'
        color_counter = color_counter + np.array([0, 0, 0, 0, 0, 1])
    else:
        color_random = '#40B0A6'
        color_counter = color_counter + np.array([1, 0, 0, 0, 0, 0])
    # CHANGE depending on what you're plotting
    plt.plot(np.array(time)*T_final*TU/86400, np.array(dx_norm)*DU/TU, color=color_random)

print(color_counter)

plt.legend(["Ref","2nd", "3rd", "4th", "5th", "6th"])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[1].set_color('#FFC0CB')
leg.legend_handles[2].set_color('#994F00')
leg.legend_handles[3].set_color('#006CD1')
leg.legend_handles[4].set_color('#E1BE6A')
leg.legend_handles[5].set_color('#FF5F1F')
#leg.legend_handles[5].set_color('#40B0A6')


#for displacement plot
plt.xlabel('$t$ [days]')
# CHANGE depending on what you're plotting
plt.ylabel('$v_z$ [m/s]')

# CHANGE depending on what you're plotting
#plt.savefig('Eigen_dVZ.pdf', format='pdf')
plt.show()







#########################
###### COLORPLOT ########
#########################
"""
fig = plt.figure()
# for 3d plot
#ax = fig.add_subplot(projection="3d")
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
color_counter = np.zeros((1,6))
for i in range(100):
    rand_vec = np.random.standard_normal(6)
    rand_vec = np.reshape(rand_vec,(6,1))
    rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
    scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
    dx_0 = rand_vec * scaling_factor
    costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
    x = np.reshape(state_full[0,:6], (6,1)) + dx_0
    dx_norm = []
    dlambda_norm = []
    for index in range(len(state_full)):
        dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
        dlambda = STM_full[index,6:12,:6] @ dx_0 + STM_full[index,6:12,6:12] @ costates_initial
        dx_norm.append(la.norm(dx[:3]))
        dlambda_norm.append(la.norm([dlambda[3:]]))
        x = np.hstack((x, np.reshape(state_full[index,:6], (6,1)) + dx))

    angles = [np.abs(np.dot(rand_vec[:,0],w[:,0])), np.abs(np.dot(rand_vec[:,0],w[:,1])), 
    np.abs(np.dot(rand_vec[:,0],w[:,2])), np.abs(np.dot(rand_vec[:,0],w[:,3])), 
    np.abs(np.dot(rand_vec[:,0],w[:,4])), np.abs(np.dot(rand_vec[:,0],w[:,5]))]

    index_max = np.argmax(angles)
    if index_max == 0:
        color_random = '#FF5F1F'
        color_counter = color_counter + np.array([1, 0, 0, 0, 0, 0])
    elif index_max == 1:
        color_random = '#FFC0CB'
        color_counter = color_counter + np.array([0, 1, 0, 0, 0, 0])
    elif index_max == 2:
        color_random = '#994F00'
        color_counter = color_counter + np.array([0, 0, 1, 0, 0, 0])
    elif index_max == 3:
        color_random = '#006CD1'
        color_counter = color_counter + np.array([0, 0, 0, 1, 0, 0])
    elif index_max == 4:
        color_random = '#E1BE6A'
        color_counter = color_counter + np.array([0, 0, 0, 0, 1, 0])
    else:
        color_random = '#40B0A6'
        color_counter = color_counter + np.array([0, 0, 0, 0, 0, 1])

    # For a 3d orbit plot
    #plt.plot(x[0,:], x[1,:], x[2,:], color=color_random, alpha=0.2)

    # For a 2d orbit plot
    #plt.plot(x[0,:], x[1,:],color=color_random, alpha=1)

    # For a displacement plot
    plt.plot(time, dx_norm,color=color_random)

print(color_counter)

plt.legend(["1st", "2nd", "3rd", "4th", "5th", "6th"])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('#FF5F1F')
leg.legend_handles[1].set_color('#FFC0CB')
leg.legend_handles[2].set_color('#994F00')
leg.legend_handles[3].set_color('#006CD1')
leg.legend_handles[4].set_color('#E1BE6A')
leg.legend_handles[5].set_color('#40B0A6')


#for displacement plot
plt.hlines(0, 0, 1, color = [0,0,0])
plt.xlabel("Time [TU]")
plt.ylabel("Thrust Magnitude [DU/TU^2]")
# for 2d orbit
#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
#plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
#plt.xlabel("X [DU]")
#plt.ylabel("Y [DU]")
# for 3d orbit
#ax.set(
#    xlabel='X [DU]',
#    ylabel='Y [DU]',
#    zlabel='Z [DU]',
#)
#plt.savefig('Thrust_Magnitude.eps', format='eps')
plt.show()
"""










"""
#########################
###### VALIDATION #######
#########################
# Improvement: Create a more robust continuation scheme

# iterate the state to improve accuracy (doesn't improve accuracy currently)
#state_ics = state_iterate(np.array(ics[:6],ndmin=2).T, 1E-3, 0.001, T_final)
#print(str(state_ics))
costates_guess = np.array([[0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000]])
cost_inherent, unused_parameter_costates = true_cost(np.array(ics[:6],ndmin=2).T, 0, 1E-13, 0.01, T_final, costates_guess)
u_inherent = np.sqrt(2*cost_inherent/T_final)

print("Inherent Cost is " + str(cost_inherent) + " DU^2/TU^2 or "+ str(T_final*0.5*u_inherent**2))
print("Inherent u is " + str(u_inherent) + " DU/TU")

dx_mag = []
J_linear = []
J_computed = []
J_true = []
costates_guess = np.array([[0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000]])
#for i in [600, 700, 800, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]:
#for i in [100000, 10000, 1000, 700, 500, 400, 300, 260, 225, 200, 175, 150, 135, 110, 100]:
for i in [500000, 50000, 5000, 1000, 800, 500]:
#for i in [5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 1000000000, 10000000000]:
#for i in [7500, 10000, 20000, 30000, 31000, 32500, 34000, 35500, 37000, 38500, 40000, 44000, 50000, 70000]:
    dx = np.array(w[:,3]) * 1/i
    dx_mag.append(la.norm(dx))
    J_linear.append(0.5*np.dot(dx, np.dot(dx,E_star))) # This line is in place of 0.5xtranspose(a)xE*xa where a is 6x1 and E* is 6x6

    # compute the true cost
    cost_computed, costates_guess = true_cost(np.array(ics[:6],ndmin=2).T + np.array(w[:,3],ndmin=2).T * 1/i, u_inherent, 1E-13, 0.01, T_final, costates_guess)
    J_computed.append(cost_computed)

J_difference_computed = 100 * abs(np.array(J_linear) - np.array(J_computed)) / np.array(J_computed)

#print(str(state[:,:6]))
#print("Differences in final and initial state are " + str(100 * (np.array(ics[:6])-np.array(state[:,:6])) / (np.array(ics[:6])+np.array(state[:,:6]))) + " %" )

dx_mag_plot = [dx_mag[0], dx_mag[1], dx_mag[2], dx_mag[-1]]
J_linear_plot = [J_linear[0], J_linear[1], J_linear[2], J_linear[-1]]
J_computed_plot = [J_computed[0], J_computed[1], J_computed[2], J_computed[-1]]
abs_error = abs(np.array(J_linear_plot) - np.array(J_computed_plot))

plt.figure()
plt.loglog(dx_mag_plot, abs_error,'X', color=[0, 0, 0])
plt.xlabel('|dx| [DU]')
plt.ylabel('Absolute Error in J (DU^2/TU^3)')
#plt.savefig('Absolute_Error.pdf', format='pdf')
plt.show()

plt.figure()
plt.loglog(dx_mag_plot, J_linear_plot, 'o', markersize=15, color=[252/255, 227/255, 3/255])
plt.loglog(dx_mag_plot, J_computed_plot, 'X', color=[0, 0, 0])
plt.legend(['Estimate', 'Computed'])
plt.xlabel('|dx| [DU]')
plt.ylabel('J (DU^2/TU^3)')
#plt.savefig('Linear_Computed_Cost.pdf', format='pdf')
plt.show()


plt.figure()
plt.loglog(dx_mag / la.norm(ics[:6]), J_difference_computed, 'bX')
plt.xlabel('|dx|/|x|')
plt.ylabel('% Relative Error in J')
plt.show()

"""
