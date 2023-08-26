import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# simulating local volume fraction variance
def get_lvfv_3d(im, no_radii=50, no_samples_per_radius = 50,
                min_radius= 2, max_radius = 100,
                phase=None, grid=True):
    
    # set a range of r 
    radii = np.linspace(min_radius,max_radius,no_radii, dtype='int')

    lvfv=[]
    for r in tqdm(radii):
        lvfv.append(vfv_3d(im, r, no_samples_per_radius, phase, grid))
    return lvfv, radii

def vfv_3d(im, r, no_samples, phase=None, grid=True):
    """
    Calculates the volume fraction variance
    """
    cntrs = get_centers_3d(im, r, no_samples,grid)
    vfs=[]
    for cntr in cntrs:
        s = get_slice_3d(im,cntr, r)    
        vf = phi(s, phase)
        vfs.append(vf)
    var = np.var(vfs)
    return var
    
def grid_points(no_points, array):
    
    """
    Gets indices of "no_points" voxels distributed in a grid within array
    
    Parameters:
        no_points: number of centeroids in the grid
        array: 3D array of the explored data. required for defining the grid geometry.
    return:
        Centroids
    """
    
    pts=1000
    x,y,z =array.shape
    size = array.size
    if pts>size:
        pts=size
    f = 1-(pts/size)
    s = np.ceil(np.array(array.shape) * f).astype('int')
    nx,ny,nz = s
    xs=np.linspace(0,x, nx, dtype='int', endpoint=True)
    ys=np.linspace(0,y, ny, dtype='int', endpoint=True)
    zs=np.linspace(0,z, nz, dtype='int', endpoint=True)

    rndx,rndy,rndz = np.meshgrid(xs,ys,zs)
    rndx = rndx.flatten()
    rndy = rndy.flatten()
    rndz = rndz.flatten()
    centers=(np.array([rndx,rndy,rndz]).T)[:pts]
    
    return centers

def get_centers_3d(im, r, no_centers, grid=True):
    
    """
    im: 3D image
    r: radius of moving window
    no_centers: number of centers for the moving windows
    
    auto: (default= True) this makes sure that all the image is covered by the moving windows
            also makes sure that the number of generated centroids is >= no_centers.
            
            when false, random coordinates are generated where the number of generated centroids = no_centers.
    
    adjust_no_centers: when true, no_centers is adjusted to save running time. 
                        So, in case of big window, the returned coordinates are <= no_centers, while windows cover all image
    
    max_no_centers: maximum number of centers to be returned. None returns all centers
    
    returns (n,3) centers for a cubic window with side = 2r
    
    """
    #-------adjust window's radius with image size------
    ss=np.array(im.shape)
    cnd = r>=ss/2
    
    if sum(cnd)==0:
        mn = np.array([r,r,r])
        mx=ss-r
    else:
        mn = (cnd*ss/2)  + np.invert(cnd) * r
        mx=(cnd*mn + cnd)+ (np.invert(cnd) * (ss-r))

    rw_mn, col_mn, z_mn = mn.astype(int)
    rw_mx, col_mx, z_mx = mx.astype(int)
    #----------------------------------------------------
    if grid: 
        centers = grid_points(no_centers, im)
        
    else:
        #------random centroids----------------------
        rndx=np.random.randint(rw_mn, rw_mx, no_centers)
        rndy=np.random.randint(col_mn,col_mx, no_centers)
        rndz=np.random.randint(z_mn,z_mx, no_centers)
        centers=np.array([rndx,rndy,rndz]).T

    return centers
    

def segmoid(x):
    return 1/(1+np.exp(-x))
                   
    
    
def get_slice_3d(im, center, r):
    
    """
    This slices the image with a cube whose center = center
    the cube's dimensions are corrected acording to the image size
    """
    #----check whether the window would exceeds the image boundaries----if so, cut window------
    rr,cc,zz=center
    mn = np.array([rr-r, cc-r, zz-r])
    mx = np.array([rr+r+1, cc+r+1, zz+r+1])
    mn = (mn>0) * mn 
    
    rw_mn, col_mn, z_mn = mn
    rw_mx, col_mx, z_mx = mx # if it exceeds the image size it doesn't change anything
    
    return im[rw_mn:rw_mx, col_mn:col_mx, z_mn:z_mx]


def phi(im, phase=None, return_fraction=False):
    """
    calculates volume fraction (e.g., porosity) of
    a phase in an image im
    """
    if phase==None: # assumes binary data
        sm=np.sum(im)
    else: 
        sm = len(im[im==phase])
    
    phi_ = sm/im.size
    if return_fraction==True:
        return phi_, sm
    else:
        return phi_




class Vsi():
    def __init__(self,im, no_radii=50,
                 no_samples_per_radius = 50, 
                 min_radius= 1, max_radius = 100, 
                 phase=None, grid=False):
        """
        This calculates porosity variance (scale independent), based on a moving window with various radii.
        
        Parameters:
        
            im: 3D segmented (with different phases labelled) or binary image (True for pores and False for solids)
            no_radii: number of radii to be used for moving windows
            no_samples_per_radius: number of windows per radius
            min_radius: minimum radius of windows
            max_radius: maximum radius of windows
            phase: label of phase of interest. default= None assuming binary with pores are ones/True
            grid: if True, the windows are distributed on a grid having same geometry as image (im). However, the number of centroids is
                controlled by "no_samples_per_radius". Inactive when auto_centers is True.

        returns
            variance & radii 
        """
        
        var = get_lvfv_3d(im, no_radii, 
                          no_samples_per_radius, 
                          min_radius, max_radius,
                          phase,grid)
    
        self.variance = var[0]
        self.radii = var[1]
        
    def plot(self, label=None, 
             x_label='relative radius', y_label='Porosity Variance',
             fill=True, fill_all=False,legend=True, local = False):
        
        """
        Plots the resulted variance
        
        Parameters:
            label: Label or name of the rock sample
            x_label & y_label: labels of the X-axis and Y-axis
            fill: whether to fill Homogeneity and Heterogeneity zones
            legend: whether to show legend or not
        """
        
        y=self.variance
        plt.plot(y, label=label)
        x=np.linspace(-2,17,len(y))
        x2=np.linspace(-2,6,len(y))
        bound = (0.023*(1-segmoid(x)))
        bnd = bound[bound<=0.0025]
        bound[bound<=0.0025] = np.linspace(0.0025,0.001,len(bnd))
        if fill:
            plt.fill_between(range(len(y)), bound,facecolor='g', alpha=0.3,label='Homogeneity Zone')
            if fill_all:
                plt.fill_between(range(len(y)), bound, 0.035,facecolor='r', alpha=0.3,label='Heterogeneity Zone')
            else:
                plt.fill_between(range(len(y)), bound, ((0.035*(1-segmoid(x2))))+0.007,facecolor='r', alpha=0.3,label='Heterogeneity Zone')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        if local:
            tks = np.linspace(0,len(y)-1, 6, dtype='int')
            lbls = self.radii[tks]
            plt.xticks(tks, lbls)
        
        if legend:
            plt.legend()
       
