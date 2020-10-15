#!/usr/bin/env python
# coding: utf-8

# In[113]:


from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().magic(u'matplotlib qt')

plt.close('all')

# Take the most recent mat file
dir_path = '/tmp' 
list_of_files = glob.glob(dir_path+'/xbot_cartesian_opensot_log_*.mat') 
file_to_load = max(list_of_files, key=os.path.getctime)

print('Loading file: ', file_to_load)

data = loadmat(file_to_load)

s0 = data['visual_servoing_camera_link_s0'] 
s1 = data['visual_servoing_camera_link_s1'] 
s2 = data['visual_servoing_camera_link_s2'] 
s3 = data['visual_servoing_camera_link_s3'] 

sd0 = data['visual_servoing_camera_link_sd0'] 
sd1 = data['visual_servoing_camera_link_sd1'] 
sd2 = data['visual_servoing_camera_link_sd2'] 
sd3 = data['visual_servoing_camera_link_sd3'] 

q = data['q']
dq = data['dq']

data_freq = 1000 # Hz

# Cut data

idxs = np.array([])
for k in range(0,6):
    idx = np.argmax(np.abs(dq[k,:])>1e-3)
    idxs = np.append(idxs,idx)

ini = int(np.min(idxs))-1

'''idxs = np.array([])
for k in range(0,len(dq)):
    idx = np.argmax(np.abs(np.diff(dq[k,ini:len(dq[k])])) <= 1e-8)
    idxs = np.append(idxs,idx)
    print(k, idx)
fin = int(np.max(idxs))
'''
fin=len(s0[0])

time_vec = np.arange(0,fin-ini,1) / data_freq

print('Orginal opensot data long '+str(len(s0[0]))+' samples')
print('Cutting opensot data from '+str(ini)+' to '+str(fin))

x0 = s0[0,ini:fin]
y0 = s0[1][ini:fin]

xd0 = sd0[0,ini:fin]
yd0 = sd0[1,ini:fin]

x1 = s1[0,ini:fin]
y1 = s1[1,ini:fin]

xd1 = sd1[0,ini:fin]
yd1 = sd1[1,ini:fin]

x2 = s2[0,ini:fin]
y2 = s2[1,ini:fin]

xd2 = sd2[0,ini:fin]
yd2 = sd2[1,ini:fin]

x3 = s3[0,ini:fin]
y3 = s3[1,ini:fin]

xd3 = sd3[0,ini:fin]
yd3 = sd3[1,ini:fin]

# It would be nice to have this from data
focal = 554
umax = 640
umin = 0
uc = 320
vmax = 480
vmin = 0
vc = 240
xmax = (umax - uc) / focal 
ymax = (vmax - vc) / focal 
xmin = (umin - uc) / focal 
ymin = (vmin - vc) / focal 

print('Plotting data...')

plt.close('all')

n_fig = 1

plt.close(n_fig)
fig = plt.figure(n_fig)

ax = fig.add_subplot(111)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

plt.axis([xmin, xmax, ymin, ymax])

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.locator_params(tight=True,nbins=5)

plt.title('Image plane')

plt.ion()
plt.show() 

plt.xlabel("x")
plt.ylabel("y")

plt.plot(x0,y0,'tab:green',linestyle='--',linewidth=1,markersize=0.9)
plt.plot(x0[0],y0[0],'tab:green',marker='.', markersize=8)

plt.plot(x1,y1,'tab:green',linestyle='--',linewidth=1,markersize=0.9)
plt.plot(x1[0],y1[0],'tab:green',marker='.',markersize=8)

plt.plot(x2,y2,'tab:green',linestyle='--',linewidth=1,markersize=0.9)
plt.plot(x2[0],y2[0],'tab:green',marker='.',markersize=8)

plt.plot(x3,y3,'tab:green',linestyle='--',linewidth=1,markersize=0.9)
plt.plot(x3[0],y3[0],'tab:green',marker='.',markersize=8)

plt.plot(xd0,yd0,'tab:red',marker='o',markerfacecolor='none',markeredgewidth=1)
plt.plot(xd1,yd1,'tab:red',marker='o',markerfacecolor='none',markeredgewidth=1)
plt.plot(xd2,yd2,'tab:red',marker='o',markerfacecolor='none',markeredgewidth=1)
plt.plot(xd3,yd3,'tab:red',marker='o',markerfacecolor='none',markeredgewidth=1)

name_fig = 'image_plane.pdf'
print('Saving ' + name_fig + ' in ' + dir_path)
plt.savefig(dir_path+'/'+name_fig)

command_sys = 'pdfcrop ' + dir_path+'/'+name_fig + ' ' +dir_path+'/'+name_fig
print(command_sys)
_ = os.system(command_sys)


# In[114]:


# VISUAL ERROR

n_fig = 123400
plt.close(n_fig)
fig = plt.figure(n_fig)

ax = fig.add_subplot(2,1,1)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

norm_e = np.sqrt((x0-xd0)**2+(x1-xd1)**2+(x2-xd2)**2+(x3-xd3)**2
                 +(y0-yd0)**2+(y1-yd1)**2+(y2-yd2)**2+(y3-yd3)**2)

xmin = time_vec[0]
xmax = time_vec[len(time_vec)-1]
ymin = np.min(norm_e)
ymax = np.max(norm_e)
off_y = 0.1 * np.abs(ymax - ymin)

plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

#plt.locator_params(tight=True,nbins=5)

plt.ion()
plt.show() 

plt.xlabel("time [s]")
plt.ylabel("$||s-s_d||$")
plt.plot(time_vec,norm_e,'tab:green',linewidth=1)

name_fig = 'visual_error.pdf'
print('Saving ' + name_fig + ' in ' + dir_path)
plt.savefig(dir_path+'/'+name_fig)

command_sys = 'pdfcrop ' + dir_path+'/'+name_fig + ' ' +dir_path+'/'+name_fig
print(command_sys)
os.system(command_sys)


# In[115]:


from scipy.spatial.transform import Rotation as R

# Take the most recent mat file
dir_path = '/tmp' 
list_of_files = glob.glob(dir_path+'/gazebo_floating_base__*.mat') 
file_to_load = max(list_of_files, key=os.path.getctime)

print('Loading file: ', file_to_load)

data_freq_gz = 1000

data_gazebo = loadmat(file_to_load)

fb_pos_gz = data_gazebo['gazebo_fb_pos']
fb_vel_gz = data_gazebo['gazebo_fb_vel']
fb_rot_gz = data_gazebo['gazebo_fb_rot']
com_pos_gz = data_gazebo['gazebo_com_pos']

fin_gz = len(fb_vel_gz[0,:])

idxs = np.array([])
for k in range(0,6):
    idx = np.argmax(np.abs(fb_vel_gz[0,:])>1e-6)
    idxs = np.append(idxs,idx)

ini_gz_zero = int(np.min(idxs)) + 3 * data_freq_gz

print(ini_gz_zero)

idxs = np.array([])
for k in range(0,6):
    idx = np.argmax(np.abs(fb_vel_gz[0,ini_gz_zero:fin_gz])>1e-6) + ini_gz_zero
    idxs = np.append(idxs,idx)

ini_gz = int(np.min(idxs)) - 1

print('Orginal opensot data long '+str(len(fb_pos_gz[0]))+' samples')
print('Cutting opensot data from '+str(ini_gz)+' to '+str(fin_gz))

time_vec_gz = np.arange(0,fin_gz-ini_gz,1) / data_freq_gz

# Build the pose at the beginning of the experiment.
rot_mat_0 = fb_rot_gz[:,:,ini_gz]
pos_0 = fb_pos_gz[:,ini_gz]
T_0 = np.block([
    [rot_mat_0, pos_0.reshape(3,1)],
    [np.array([0, 0, 0, 1])]
])   

fb_rot_vec0_gz = np.empty((0,3), int)
fb_pos_vec0_gz = np.empty((0,3), int)

print('Initialize pose w.r.t. the beginning of the movement (it takes some time...)')
for k in range(ini_gz,fin_gz):
    
    rot_mat = fb_rot_gz[:,:,k]
    pos = fb_pos_gz[:,k]
    
    # Get the current pose
    T = np.block([
        [rot_mat, pos.reshape(3,1)],
        [np.array([0, 0, 0, 1])]
    ])
    
    # Transform the pose w.r.t. the begining of the experiment
    T_ = np.matmul(np.linalg.inv(T_0),T)
    
    rot_mat_ = T_[0:3,0:3]
    r = R.from_matrix(rot_mat_)
    rot_vec = r.as_euler('zyx', degrees=False)
    fb_rot_vec0_gz = np.append(fb_rot_vec0_gz,[np.array([rot_vec[2],rot_vec[1],rot_vec[0]])],axis=0)
    
    pos_ = T_[0:3,3]
    fb_pos_vec0_gz = np.append(fb_pos_vec0_gz,[pos_],axis=0)
    
print('Done.')

#com_gz = data_gazebo['gazebo_com_pos']


# In[110]:


plt.close('all')
plt.plot(fb_vel_gz[0,:])
#plt.plot(fb_pos_gz[1,:])
#plt.plot(fb_pos_gz[2,:])

#plt.plot(q[0,:])


# In[116]:


# CONFIGURATION

n_fig = 2#+=1
plt.close(n_fig)
fig_q = plt.figure(n_fig)

ax = fig_q.add_subplot(311)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

plt.ion()
plt.show()

xmin = time_vec[0]
xmax = time_vec[-1]
ymin = min(q[0:3].flatten())
ymax = max(q[0:3].flatten())
off_y = 0.1 * np.abs(ymax - ymin)

plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

plt.ylabel("FB position [m]")

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=None)

ax.xaxis.set_ticklabels([])

coord_colors = np.array(['tab:red','tab:green','tab:blue'])
labels_pose = np.array(['x (QP)','y (QP)','z (QP)','roll (QP)','pitch (QP)','yaw (QP)'])

for k in range(0,3):
    #print(k)
    plt.plot(time_vec,q[k][ini:fin],coord_colors[k],linewidth=1,label=labels_pose[k])

plt.plot(time_vec_gz,fb_pos_vec0_gz[:,0],'tab:red',linestyle='--',linewidth=1,label='x (GT)')
plt.plot(time_vec_gz,fb_pos_vec0_gz[:,1],'tab:green',linestyle='--',linewidth=1,label='y (GT)')
plt.plot(time_vec_gz,fb_pos_vec0_gz[:,2],'tab:blue',linestyle='--',linewidth=1,label='z (GT)')

handles,labels = ax.get_legend_handles_labels()

handles = [handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]]
labels = [labels[0], labels[3], labels[1], labels[4], labels[2], labels[5]]

ax.legend(handles,labels,fontsize='x-small',ncol=3,loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

#ax.legend(['x (QP)','y (QP)','z (QP)',
#           'x (GT)','y (GT)','z (GT)'],
#          fontsize='x-small',ncol=3,loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

ax = fig_q.add_subplot(312)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

ymin = min(q[3:6].flatten())
ymax = max(q[3:6].flatten())
off_y = 0.1 * np.abs(ymax - ymin)

plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

plt.ylabel("FB orient. [rad]")
ax.xaxis.set_ticklabels([])

for k in range(3,6):
    #print(k)
    plt.plot(time_vec,q[k][ini:fin],coord_colors[k-3],linewidth=1,label=labels_pose[k])

plt.plot(time_vec_gz,fb_rot_vec0_gz[:,0],'tab:red',linestyle='--',linewidth=1,label='roll (GT)')
plt.plot(time_vec_gz,fb_rot_vec0_gz[:,1],'tab:green',linestyle='--',linewidth=1,label='pitch (GT)')
plt.plot(time_vec_gz,fb_rot_vec0_gz[:,2],'tab:blue',linestyle='--',linewidth=1,label='yaw (GT)')

handles,labels = ax.get_legend_handles_labels()

handles = [handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]]
labels = [labels[0], labels[3], labels[1], labels[4], labels[2], labels[5]]

ax.legend(handles,labels,fontsize='x-small',ncol=3,loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
'''
ax.legend(['roll (QP)','pitch (QP)','yaw (QP)', 
           'roll (GT)','pitch (GT)','yaw (GT)'],
          fontsize='x-small',ncol=3, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
'''
ax = fig_q.add_subplot(313)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

ymin = min(q[6:len(q)].flatten())
ymax = max(q[6:len(q)].flatten())
off_y = 0.1 * np.abs(ymax - ymin)

plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

plt.ylabel('$q_a$ [rad]')
plt.xlabel("time [s]")

for k in range(6,len(q)):
    #print(k)
    if np.any(np.abs(dq[k][ini:fin])>0.0):
        plt.plot(time_vec,q[k][ini:fin],linewidth=1)
    else:
        print('Joint n. '+str(k)+' does not move')

fig_q.align_ylabels()

name_fig = 'configuration.pdf'
print('Saving ' + name_fig + ' in ' + dir_path)
plt.savefig(dir_path+'/'+name_fig)

command_sys = 'pdfcrop ' + dir_path+'/'+name_fig + ' ' +dir_path+'/'+name_fig
print(command_sys)
os.system(command_sys)


# In[96]:


# COM AND CENTROIDAL MOMENTUM 

import h5py
    
list_of_files = glob.glob(dir_path+'/cartesio_logger__*.mat') 
file_to_load = max(list_of_files, key=os.path.getctime)

print('Loading file: ', file_to_load)
    
data2 = h5py.File(file_to_load, 'r')

h_ = data2['ci_centroidal_momentum'][()]
h = np.transpose(h_)

com_ = data2.get('Com_pos')[()]
com = np.transpose(com_)

ini = np.argmax(np.abs(h[:])>1e-10) - 1
fin = len(h[0])

time_vec = np.arange(0,fin-ini,1) / data_freq

print('Cutting data from '+str(ini)+' to '+str(fin))

n_fig = 3#+=1
plt.close(n_fig)
fig_q = plt.figure(n_fig)

#com = data['CoM_pos_actual']

# Centroidal momentum

ax = fig_q.add_subplot(4,1,(1,2))

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

lin_mom_norm = np.sqrt(h[0][ini:fin]**2+h[1][ini:fin]**2+h[2][ini:fin]**2)
ang_mom_norm = np.sqrt(h[3][ini:fin]**2+h[4][ini:fin]**2+h[5][ini:fin]**2)

xmin = time_vec[0]
xmax = time_vec[-1]
ymin = np.min(np.array([lin_mom_norm.flatten(),ang_mom_norm.flatten()]))
ymax = np.max(np.array([lin_mom_norm.flatten(),ang_mom_norm.flatten()]))

off_y = 0.1 * np.abs(ymax - ymin)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=None)

print([xmin, xmax, ymin - off_y, ymax + off_y])
plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

plt.ylabel("Norm of CM [N s, N m s]")

ax.xaxis.set_ticklabels([])

lin,=plt.plot(time_vec,lin_mom_norm,'tab:olive',linewidth=1)

'''lin, = plt.plot(time_vec,h[0][ini:end],'tab:blue',linewidth=1)
plt.plot(time_vec,h[1][ini:end],'tab:blue',linewidth=1)
plt.plot(time_vec,h[2][ini:end],'tab:blue',linewidth=1)'''

ang, = plt.plot(time_vec,ang_mom_norm,'tab:gray', linestyle='-',linewidth=1)
'''ang, = plt.plot(time_vec,h[3][ini:end],'tab:blue', linestyle='--',linewidth=1)
plt.plot(time_vec,h[4][ini:end],'tab:blue', linestyle='--',linewidth=1)
plt.plot(time_vec,h[5][ini:end],'tab:blue', linestyle='--',linewidth=1)'''

plt.legend([lin, ang], ['linear', 'angular'], fontsize='x-small',ncol=1)

# COM 

ax = fig_q.add_subplot(4,1,3)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

plt.ion()
plt.show()

xmin = time_vec[0]
xmax = time_vec[-1]
ymin = min(com.flatten())
ymax = max(com.flatten())
off_y = 0.1 * np.abs(ymax - ymin)

plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

plt.ylabel("CoM [m]")
plt.xlabel("time [s]")

offset = com_pos_gz[:,0] - com[:,0]

lbs = np.array(['x (QP)','y (QP)','z (QP)','x (GT)','y (GT)','z (GT)'])

for k in range(0,3):
    #print(k)
    plt.plot(time_vec,com[k,ini:fin],coord_colors[k],linewidth=1,label=lbs[k])
    plt.plot(time_vec_gz,com_pos_gz[k,ini_gz:fin_gz]-offset[k],coord_colors[k],linestyle='--',linewidth=1, label=lbs[k+3])


handles,labels = ax.get_legend_handles_labels()

ax.legend(handles,labels,fontsize='x-small',ncol=3,loc='best')#, bbox_to_anchor=(0.5, 0., 0.5, 0.5))

'''ax = fig_q.add_subplot(313)

plt.grid(linestyle='-',linewidth=1,color='whitesmoke')

ymin = min(q[6:len(q)].flatten())
ymax = max(q[6:len(q)].flatten())
off_y = 0.1 * np.abs(ymax - ymin)

plt.axis([xmin, xmax, ymin - off_y, ymax + off_y])

plt.ylabel('Lin. mom. [rad]')
plt.xlabel("iterations")

for k in range(6,len(q)):
    #print(k)
    plt.plot(k,k*k,linewidth=1)'''

fig_q.align_ylabels()


name_fig = 'CoM_CM.pdf'
print('Saving ' + name_fig + ' in ' + dir_path)
plt.savefig(dir_path+'/'+name_fig)

command_sys = 'pdfcrop ' + dir_path+'/'+name_fig + ' ' +dir_path+'/'+name_fig
print(command_sys)
os.system(command_sys)


# In[55]:


plt.close('all')
plt.plot(fb_vel_gz[0,:])
plt.show()


# In[38]:


com_pos_gz[0,0] - offset[0]


# In[10]:


plt.plot(np.linalg.norm(h[0:3],ord=2,axis=0), 'k')
plt.plot(np.sqrt(h[0]**2+h[1]**2+h[2]**2),'r--')


# In[11]:



# 3D PLOT

from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial.transform import Rotation as R

def plot_camera(P,ax,color_scale):

    P.resize(5,4)

    # Plot image plane
    x_coord = np.array([P[1][0],P[2][0],P[3][0],P[4][0],P[1][0]])
    y_coord = np.array([P[1][1],P[2][1],P[3][1],P[4][1],P[1][1]])
    z_coord = np.array([P[1][2],P[2][2],P[3][2],P[4][2],P[1][2]])

    '''
    rect = Rectangle((-0.2,-0.15), 0.4, 0.3, facecolor="whitesmoke", alpha=0.2)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="x")
    '''
    
    # Wire model 
    
    color = (color_scale,color_scale,color_scale)
    
    linewidth = 1
    
    ax.plot(x_coord, y_coord, z_coord, color=color, linewidth=linewidth)

    ax.plot(np.array([P[0][0],P[1][0]]),np.array([P[0][1],P[1][1]]),np.array([P[0][2],P[1][2]]),color=color,linewidth=linewidth)
    ax.plot(np.array([P[0][0],P[2][0]]),np.array([P[0][1],P[2][1]]),np.array([P[0][2],P[2][2]]),color=color,linewidth=linewidth)
    ax.plot(np.array([P[0][0],P[3][0]]),np.array([P[0][1],P[3][1]]),np.array([P[0][2],P[3][2]]),color=color,linewidth=linewidth)
    ax.plot(np.array([P[0][0],P[4][0]]),np.array([P[0][1],P[4][1]]),np.array([P[0][2],P[4][2]]),color=color,linewidth=linewidth)

def camera_pose(T):
    
    scale = 0.1
    
    p0 = scale * np.array([-3, 0,   0, 1])
    p1 = scale * np.array([ 0,-2, 1.5, 1])
    p2 = scale * np.array([ 0, 2, 1.5, 1])
    p3 = scale * np.array([ 0, 2,-1.5, 1])
    p4 = scale * np.array([ 0,-2,-1.5, 1])
    
    P = np.array([])
    
    p0w = np.matmul(T,p0)
    P = np.append(P,p0w)
    
    p1w = np.matmul(T,p1)
    P = np.append(P,p1w)
    
    p2w = np.matmul(T,p2)
    P = np.append(P,p2w)
    
    p3w = np.matmul(T,p3)
    P = np.append(P,p3w)
    
    p4w = np.matmul(T,p4)
    P = np.append(P,p4w)
    
    return P

def plot_visual_pattern(X,Y,Z,ax):
    
    offset = 0.05
    
    linewidth = 1
    
    ax.plot(np.append(Z,Z[0]),-np.append(X,X[0]),-np.append(Y,Y[0]),'r.-')
    
    '''
    ax.plot(np.append(Z,Z[0]),
            -np.array([X[0]-offset,X[1]+offset,X[2]+offset,X[3]-offset,X[0]-offset]),
            -np.array([Y[0]-offset,Y[1]-offset,Y[2]+offset,Y[3]+offset,Y[0]-offset]),
            'k',linewidth=linewidth
           )
    ax.plot(np.append(Z_vp,Z_vp[0])+0.02,
            -np.array([X[0]-offset,X[1]+offset,X[2]+offset,X[3]-offset,X[0]-offset]),
            -np.array([Y[0]-offset,Y[1]-offset,Y[2]+offset,Y[3]+offset,Y[0]-offset]),
            'k'
           )    
    '''

n_fig = n_fig + 1
plt.close(n_fig)
fig = plt.figure(n_fig)

ax = fig.add_subplot(111, projection='3d')

plt.ion()
plt.show() 

ax.locator_params(nbins=3)

ax.set_zlim3d(-2.0,3.0)
ax.set_ylim3d(-4.0,1.0)
ax.set_xlim3d(0,5)

#ax.auto_scale_xyz([0, 2], [-1, 1], [-1, 1])

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.w_xaxis.gridlines.set_lw(3.0)
ax.w_yaxis.gridlines.set_lw(3.0)
ax.w_zaxis.gridlines.set_lw(3.0)

ax.w_xaxis.gridlines.set_color('white')
#ax.w_yaxis.gridlines.set_lw(3.0)
#ax.w_zaxis.gridlines.set_lw(3.0)

#ax.axes.grid(color='r', linestyle='-', linewidth=2)

ax.view_init(elev=35, azim=135)

for k in range(0,len(q[0][ini:fin]),100):
    
    pos = np.array([q[0][k],q[1][k],q[2][k]])
    rot = np.array([q[3][k],q[4][k],q[5][k]])
    
    r = R.from_rotvec(rot)
    Rot = r.as_matrix()
        
    T = np.block([
        [Rot, pos.reshape(3,1)],
        [np.array([0, 0, 0, 1])]
    ])

    P = camera_pose(np.linalg.inv(T))
    
    color_scale = 1 - k / len(q[0][ini:fin])
    
    plot_camera(P,ax,color_scale)
    
depth = 4.5

X_vp = np.array([depth*xd0[-1], depth*xd1[-1], depth*xd2[-1],depth* xd3[-1]])
Y_vp = np.array([depth*yd0[-1], depth*yd1[-1], depth*yd2[-1],depth* yd3[-1]])
Z_vp = np.array([depth, depth, depth,depth])

plot_visual_pattern(X_vp,Y_vp,Z_vp,ax)

#plt.savefig('3D.pdf')


# In[ ]:




