import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from numpy import sin, cos

size = 50
arr_step = 2.5
normalize = True

animating = False

S = np.arange(0,size+arr_step,arr_step)
Z = np.arange(0,size+arr_step,arr_step)

pi = 1
psi = 0.1
delta = 0.1
beta = 0.1
alpha = 0.15
zeta = 0.1

def get_d_field(alph=alpha):
	#assumes zeta*R*Z >> delta*S, so all removed immediately zombify
	U = np.array([[(pi - psi - delta)*x_i - beta*x_i*y_i for x_i in S] for y_i in Z])
	V = np.array([[(beta - alph)*x_i*y_i + delta*x_i for x_i in S] for y_i in Z])
	if normalize:
		N_prelim = np.sqrt(U**2+V**2)*0.5
		N = [[1 if el==0 else el for el in row] for row in N_prelim]
		U, V = U/N, V/N
	return U,V


fig, ax = plt.subplots()

def get_trajectory(S0,Z0,step=0.01,iters=500,alph=alpha, s_ub=size, z_ub = size, use_removed=True):
	S_vals = [S0]
	Z_vals = [Z0]
	R_vals = [0]
	while S_vals[-1]>=0 and Z_vals[-1]>=0 and len(S_vals)<iters and S_vals[-1] <= s_ub and Z_vals[-1] <= z_ub:
		ls = S_vals[-1]
		lz = Z_vals[-1]
		lr = R_vals[-1]
		S_vals.append(ls + ((pi - psi - delta)*ls - beta*ls*lz)*step)
		Z_vals.append(lz + ((beta - alph)*ls*lz + (zeta*lz*lr if use_removed else delta*ls))*step)
		R_vals.append(max(lr + (delta*ls - zeta*lz*lr)*step,0))
	return S_vals, Z_vals

def get_inverse_trajectory(S0,Z0,step=0.01,iters=500,alph=alpha, s_ub=size, z_ub = size, use_removed = True, R0=0):
	S_vals = [S0]
	Z_vals = [Z0]
	R_vals = [R0]
	while S_vals[-1]>=0 and Z_vals[-1]>=0 and len(S_vals)<iters and S_vals[-1] <= s_ub and Z_vals[-1] <= z_ub:
		ls = S_vals[-1]
		lz = Z_vals[-1]
		lr = R_vals[-1]
		S_vals.append(ls - ((pi - psi - delta)*ls - beta*ls*lz)*step)
		Z_vals.append(lz - ((beta - alph)*ls*lz + (zeta*lz*lr if use_removed else delta*ls))*step)
		R_vals.append(max(lr + (delta*ls - zeta*lz*lr),0))
	return S_vals, Z_vals

def init():
	ax.set_xlabel('S')
	ax.set_ylabel('Z')
	ax.set_xlim(-size/10,size)
	ax.set_ylim(-size/10,size)

def get_data(t=0):

	t_lim = 10 #seconds of animation
	t_step = 0.5

	loop_count = t_lim/t_step

	start_alpha = 0.5*beta
	end_alpha = 2*beta

	da = (end_alpha - start_alpha)/loop_count

	count = 0
	while t < t_lim:
		t += t_step
		count += 1
		yield start_alpha + da*count

def update(alph):
	alpha = alph
	U,V = get_d_field(alpha)
	q.set_UVC(U,V)

	for s0 in range(10,40,3):
		for z0 in range(10,40,3):
			s_vals, z_vals = get_trajectory(s0,z0,alph=alph,iters=10000, use_removed=False)
			plt.plot(s_vals,z_vals,color='red')

	null_1x = np.arange(-size/10, size+2, ((size+2)-(-size/10))/1000)
	null_1y = np.array([delta/(alph-beta) for x in null_1x])
	plt.plot(null_1x, null_1y, color='green')

	null_2x = np.arange(-size/10, size+2, ((size+2)-(-size/10))/1000)
	null_2y = np.array([(pi-psi-delta)/beta for x in null_2x])
	plt.plot(null_2x, null_2y, color='green')

	#for r_end in np.arange(0,20,2):
	basin_separator_x, basin_separator_y = get_inverse_trajectory(1, (pi-psi-delta)/beta, alph=alph, iters=100000, use_removed=False, R0=2)
	plt.plot(basin_separator_x, basin_separator_y, color='blue')

	'''
	s0,z0 = 10,10
	s_vals, z_vals = get_trajectory(s0,z0,alph=alph)
	plt.plot(s_vals,z_vals,color='red')

	s0,z0 = 10,40
	s_vals, z_vals = get_trajectory(s0,z0)
	plt.plot(s_vals,z_vals,color='red')

	s0,z0 = 25,15
	s_vals, z_vals = get_trajectory(s0,z0)
	plt.plot(s_vals,z_vals,color='blue')

	s0,z0 = 25,35
	s_vals, z_vals = get_trajectory(s0,z0)
	plt.plot(s_vals,z_vals,color='blue')

	s0,z0 = 40,10
	s_vals, z_vals = get_trajectory(s0,z0)
	plt.plot(s_vals,z_vals,color='green')

	s0,z0 = 40,40
	s_vals, z_vals = get_trajectory(s0,z0)
	plt.plot(s_vals,z_vals,color='green')
	'''

	ax.annotate('$\\alpha = $'+str(round(alph,3)), xy=(0.8,0.9), xycoords='axes fraction', fontsize=16,
				bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))

	fig.canvas.draw()





U,V = get_d_field()
q = ax.quiver(S,Z,U,V)


if animating:
	start_alpha = 0.8*beta
	end_alpha = 1.5*beta

	ani = anim.FuncAnimation(fig, update, frames=np.linspace(start_alpha,end_alpha,40), blit=False, interval=10, repeat=False, init_func=init) 

else:
	init()
	update(alpha)


plt.show()
