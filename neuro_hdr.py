## CODE

#### Imports & Global Settings

import numpy as np
from scipy import signal
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator
from PIL import Image
from datetime import datetime
import streamlit as st
import psutil
import os
import gc

MAX_ENTRIES = 1

def log_memory(ref_id):
    pid = os.getpid()
    mem = psutil.Process(pid).memory_info()[0]/float(2**20)
    virt = psutil.virtual_memory()[3]/float(2**20)
    swap = psutil.swap_memory()[1]/float(2**20)

    print(f'[{datetime.now().isoformat()}]  [{ref_id}|{pid}]    rss: {mem:.2f} MB  (virtual: {virt:.2f} MB, swap: {swap:.2f} MB)')

#### Array Helper Functions

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
def float32_to_uint8(array):
    array = (255 * array).astype(np.uint8)
    return array

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)    
def uint8_to_float32(array):
    array = array.astype(np.float32) / float32(255)
    return array


@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def array_info(array, print_info=True, return_info=False, return_info_str=False):

    info = {}
    info['dtype'] = array.dtype
    info['ndim'] = array.ndim
    info['shape'] = array.shape
    info['max'] = array.max()
    info['min'] = array.min()
    info['mean'] = array.mean()
    info['std'] = array.std()
    info['size'] = array.size
    info['nonzero'] = np.count_nonzero(array)
    info['layer_variation'] = 0
    info['entropy'] = entropy(array)

    if array.ndim > 2:
        info['layer_variation'] = array.std(axis=array.ndim-1).mean()

    info['pct'] = 100 * info['nonzero'] / info['size']

    if print_info:
        print('{dtype}  {shape}'.format(**info))
        print('nonzero: {nonzero} / {size}  ({pct:.1f} %)'.format(**info))
        print('min:  {min:.2f}   max: {max:.2f}'.format(**info))
        print('mean: {mean:.2f}   std: {std:.2f}'.format(**info), end="")
        if array.ndim > 2:
            print('     layer_variation: {layer_variation:.2f}'.format(**info))

        print('entropy: {entropy:.2f}'.format(**info), end="")

    out = []
    if return_info:
        out.append(info)
    if return_info_str:
        info_str = f'shape: {info["shape"]}\n'
        info_str += f'size: {info["size"]}\nnonzero: {info["nonzero"]}  ({info["pct"]:.4f} %)\n'
        info_str += f'min: {info["min"]}    max: {info["max"]}\n'
        info_str += f'mean: {info["mean"]:.4f}    std: {info["std"]:.4f}\n'
        if array.ndim > 2:
            info_str += f'layer_variation: {info["layer_variation"]:.4f}\n'

        info_str += f'entropy: {info["entropy"]:.4f}\n'

        out.append(info_str)

    return out

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def diff(x, axis=0):
    if axis==0:
        return x[1:,:]-x[:-1,:]
    else:
        return x[:,1:]-x[:,:-1]

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def cyclic_diff(x,axis=0):
    if axis==0:
        return x[0,:]-x[-1,:]
    else:
        return (x[:,0]-x[:,-1])[None,:].T

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def flatten_by_cols(x):
    return x.flatten(order='F')
    #return x.T.reshape(np.prod(x.shape), -1).flatten()

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def flatten_by_rows(x):
    return x.flatten(order='C')
    #return x.reshape(-1, np.prod(x.shape)).flatten()

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def geometric_mean(image):
    try:
        assert image.ndim == 3, 'Warning: Expected a 3d-array.  Returning input as-is.'
        return np.power(np.prod(image, axis=2), 1/3)
    except AssertionError as msg:
        print(msg)
        return image

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def autoscale_array(array):

    array = array.astype(np.float32)
    lo = array.flatten().min()
    array -= lo    
    hi = array.flatten().max()
    try:
        assert hi > 0, f'autoscale_array cannot map null array to interval [0,1]'
    except AssertionError as msg:
        print(msg)
        return array

    array /= hi
 
    return array

def autoscale_arrays(A, B):

    lo = np.hstack([A,B]).flatten().min()
    
    A = A - lo
    B = B - lo

    hi = np.hstack([A,B]).flatten().max()
    A = A / hi
    B = B / hi
    return A, B

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def imresize(image, scale=-1, size=(-1,-1)):
    ''' image: numpy array with shape (n, m) or (n, m, 3)
       scale: mulitplier of array height & width (if scale > 0)
       size: (num_rows, num_cols) 2-tuple of ints > 0 (only used if scale <= 0)'''
    
    if (image.shape == size) | (scale == 1):
        return image

    if image.ndim==2:
        im = Image.fromarray(image)
        if scale > 0:
            width, height = im.size
            newsize = (int(width*scale), int(height*scale))
        else:
            newsize = (size[1],size[0])    #         numpy index convention is reverse of PIL

        return np.array(im.resize(newsize))

    if scale > 0:
        height = np.max(1,image.shape[0])
        width = np.max(1,image.shape[1])
        newsize = (int(width*scale), int(height*scale))
    else:
        newsize = (size[1],size[0])    

    tmp = np.zeros((newsize[1],newsize[0],3))
    for i in range(3):
        im = Image.fromarray(image[:,:,i])
        tmp[:,:,i] = np.array(im.resize(newsize))

    return tmp


#### Texture Functions
@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def delta(x):   
    
    dt0_v = np.vstack([diff(x, axis=0),cyclic_diff(x,axis=0)])
    dt0_h = np.hstack([diff(x,axis=1),cyclic_diff(x,axis=1)])
    return dt0_v, dt0_h

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def kernel(dt0_v, dt0_h, sigma):
    try:
        assert sigma%2==1, f'Warning: sigma should be odd. Using sigma = {sigma + 1}.'
    except AssertionError as warning:
        sigma += 1
        print('***** '+warning+' *****')
    n_pad = int(sigma/2)
    
    kernel_v = signal.convolve(dt0_v, np.ones((sigma,1)), method='fft')[n_pad:n_pad+dt0_v.shape[0],:]
    kernel_h = signal.convolve(dt0_h, np.ones((1,sigma)), method='fft')[:,n_pad:n_pad+dt0_h.shape[1]]
    
    return kernel_v, kernel_h

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def textures(dt0_v, dt0_h, kernel_v, kernel_h, sharpness):

    W_v = 1/(np.abs(kernel_v) * np.abs(dt0_v) + sharpness)
    W_h = 1/(np.abs(kernel_h) * np.abs(dt0_h) + sharpness)

    return W_v, W_h


#### Illumination Map Function 

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def construct_map(wx, wy, lamda):
    
    r, c = wx.shape        
    k = r * c
    
    dx = -lamda * flatten_by_cols(wx) 
    dy = -lamda * flatten_by_cols(wy)
    
    wx_permuted_cols = np.roll(wx,1,axis=1)
    dx_permuted_cols = -lamda * flatten_by_cols(wx_permuted_cols)
    
    wy_permuted_rows = np.roll(wy,1,axis=0)
    dy_permuted_rows = -lamda * flatten_by_cols(wy_permuted_rows)

    A = 1 - (dx + dy + dx_permuted_cols + dy_permuted_rows)
        
    wx_permuted_cols_head = np.zeros_like(wx_permuted_cols) 
    wx_permuted_cols_head[:,0] = wx_permuted_cols[:,0]
    dx_permuted_cols_head = -lamda * flatten_by_cols(wx_permuted_cols_head)
    
    wy_permuted_rows_head = np.zeros_like(wy_permuted_rows)
    wy_permuted_rows_head[0,:] = wy_permuted_rows[0,:]
    dy_permuted_rows_head = -lamda * flatten_by_cols(wy_permuted_rows_head)

    wx_no_tail = np.zeros_like(wx)
    wx_no_tail[:,:-1] = wx[:,:-1]
    dx_no_tail = -lamda * flatten_by_cols(wx_no_tail)

    wy_no_tail = np.zeros_like(wy)
    wy_no_tail[:-1,:] = wy[:-1,:]
    dy_no_tail = -lamda * flatten_by_cols(wy_no_tail)
    
    Ax = spdiags([dx_permuted_cols_head, dx_no_tail], [-k+r, -r], k, k)  
    
    Ay = spdiags([dy_permuted_rows_head, dy_no_tail], [-r+1,-1],  k, k)
    
    d = spdiags(A, 0, k, k)
    
    A = Ax + Ay
    A = A + A.T + d

    return A


#### Sparse solver function

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def solver_sparse(A, B, method='direct', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50):
    """
    Solves for x = b/A  [[b is vector(B)]]
    A can be sparse (csc or csr) or dense
    b must be dense
    
   """
    N = A.shape[0]
    if method == 'cg':
        if CG_prec == 'ILU':
            # Find ILU preconditioner (constant in time)
            A_ilu = spilu(A.tocsc(), drop_tol=LU_TOL, fill_factor=FILL)  
            M = LinearOperator(shape=(N, N), matvec=A_ilu.solve)
        else:
            M = None
        x0 = np.random.random(N) # Start vector is uniform
        return cg(A, B.flatten(order='F'), x0=x0, tol=CG_TOL, maxiter=MAX_ITER, M=M)[0].astype(np.float32)

        if info==MAX_ITER:
            print(f'Warning: cg max iterations ({MAX_ITER}) reached without convergence')
        
    elif method == 'direct':
        return spsolve(A, B.flatten(order='F')).astype(np.float32)
                

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def solve_linear_equation(G, A, method='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50):

    r, c = G.shape
    return solver_sparse(A,G.flatten(order='F'), method, CG_prec, CG_TOL, LU_TOL, MAX_ITER, FILL).astype(np.float32).reshape(c,r).T

    
#### Exposure Functions
@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def applyK(G, k, a=-0.3293, b=1.1258, verbose=False):
    log_memory('applyK||B')
    if k==1.0:
        return G.astype(np.float32)

    if k<=0:
        return np.ones_like(G, dtype=np.float32)

    gamma = k**a
    beta = np.exp((1-gamma)*b)

    if verbose:
        print(f'a: {a:.4f}, b: {b:.4f}, k: {k:.4f}, gamma: {gamma:.4f}, beta: {beta}.  ----->  output = {beta:.4} * image^{gamma:.4f}')

    return (np.power(G,gamma)*beta).astype(np.float32)


@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def entropy(array, bins=255, lo=0, hi=255):

    if array.dtype.name[:5] == 'float':
        array = (array * 255).astype(np.uint8)
    
    counts = np.histogram(array,bins=bins, range=(lo,hi))[0]
    counts = counts[counts>0]
    N = counts.sum()
    return (np.log2(N) - (np.dot(counts, np.log2(counts)) / N)).astype(np.float32)
    
@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def xentropy(p, q, bins=255, lo=0, hi=255):
    if p.dtype.name[:5] == 'float':
        p = (p * 255).astype(np.uint8)

    if q.dtype.name[:5] == 'float':
        q = (q * 255).astype(np.uint8)
    
    counts_p = np.histogram(p,bins=bins, range=(lo,hi))[0] + 1e-5
    counts_q = np.histogram(q,bins=bins, range=(lo,hi))[0] + 1e-5
    N = counts_q.sum()
    return (np.log2(N) - (np.dot(counts_p, np.log2(counts_q)) / N)).astype(np.float32)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def KL(P,Q, bins=255, lo=0, hi=255):
    return xentropy(P,Q, bins=bins, lo=lo, hi=hi) - entropy(P, bins=bins, lo=lo, hi=hi)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def bits2U16(bit_hot_array, mask=None):
    base2=2**np.arange(16, dtype=np.uint16)[::-1]
    if mask is not None:
        base2=base2*mask
    return np.dot(bit_hot_array, base2)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def bit_split(array):
    return np.unpackbits(array).reshape(*array.shape, 8)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def bit_concat(p, q):
    if p.dtype != 'uint8':
        if p.dtype.name[:5] == 'float':
            p = 255 * p
        p = p.astype(np.uint8)
    if q.dtype != 'uint8':
        if q.dtype.name[:5] == 'float':
            q = 255 * q
        q = q.astype(np.uint8)
        
    p_ = bit_split(p.flatten())
    q_ = bit_split(q.flatten())
    pq = np.hstack([p_,q_])
    return bits2U16(pq)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def joint_entropy(p,q, bins=256*256, lo=0, hi=256*256):
    pq = bit_concat(p,q)
    pq_counts = np.histogram(pq, bins=bins, range=(lo,hi))[0]
    del pq
    gc.collect()
    pq_counts = pq_counts[pq_counts>0]
    N = pq_counts.sum()
    return (np.log2(N) - (np.dot(pq_counts, np.log2(pq_counts)) / N)).astype(np.float32)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def mutual_information(p,q):
    return entropy(p) + entropy(q) - joint_entropy(p,q)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def variation_of_information(p,q):
    return joint_entropy(p,q) - mutual_information(p,q)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def normalized_variation_of_information(p,q):
    joint = joint_entropy(p,q) + 1e-5
    mutual = mutual_information(p,q)
    return (1. - mutual/joint)

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def conditional_entropy(p,q):
    return entropy(p) - mutual_information(p,q)


@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def get_dim_pixels(image,dim_pixels,dim_size=(50,50)):
    
    dim_pixels = imresize(dim_pixels,size=dim_size)

    image = imresize(image,size=dim_size)
    image = np.where(image>0,image,0)
    Y = geometric_mean(image)
    return Y[dim_pixels]

@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def optimize_exposure_ratio(array, a, b, lo=1, hi=7, npoints=20):
  
    if sum(array.shape)==0:
        return 1.0

    sample_ratios = np.r_[lo:hi:np.complex(0,npoints)].tolist()
    entropies = np.array(list(map(lambda k: entropy(applyK(array, k, a, b)), sample_ratios)))
    optimal_index = np.argmax(entropies)
    return sample_ratios[optimal_index]



@st.cache(max_entries=MAX_ENTRIES, show_spinner=False)
#@profile
def bimef(image, exposure_ratio=-1, enhance=0.5, 
          a=-0.3293, b=1.1258, lamda=0.5, 
          sigma=5, scale=0.3, sharpness=0.001, 
          dim_threshold=0.5, dim_size=(50,50), 
          solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, 
          lo=1, hi=7, npoints=20,
          verbose=False, print_info=True):

    log_memory('bimef||B')
    
    tic = datetime.now()

    log_memory('bimef|image_maxRGB|B')
    if image.ndim == 3: 
        image = np.copy(image[:,:,:3])
        image_maxRGB = image.max(axis=2)
    else: image_maxRGB = np.copy(image)
    log_memory('bimef|image_maxRGB|E')

    log_memory('bimef|scale|B')
    if (scale <= 0) | (scale >= 1) : image_maxRGB_reduced = image_maxRGB
    else: image_maxRGB_reduced = imresize(image_maxRGB, scale)
    log_memory('bimef|scale|E')

    log_memory('bimef|autoscale_array|B')
    image_maxRGB_reduced_01 = autoscale_array(image_maxRGB_reduced)
    log_memory('bimef|autoscale_array|E')
    
    ############ TEXTURE MAP  ###########################
    log_memory('bimef|delta|B')
    dt0_v, dt0_h = delta(image_maxRGB_reduced_01)
    log_memory('bimef|delta|E')
    log_memory('bimef|kernel|B')
    kernel_v, kernel_h = kernel(dt0_v, dt0_h, sigma)
    log_memory('bimef|kernel|E')
    log_memory('bimef|textures|B')
    wx, wy = textures(dt0_v, dt0_h, kernel_v, kernel_h, sharpness)
    log_memory('bimef|textures|E')
    ######################################################

    ############ ILLUMINATION MAP  ###########################
    log_memory('bimef|construct_map|B')
    illumination_map = construct_map(wx, wy, lamda) 
    log_memory('bimef|construct_map|E')
    ######################################################

    ############ SOLVE LINEAR EQUATION:  ###########################
    log_memory('bimef|solve_linear_equation|B')
    image_maxRGB_reduced_01_smooth = solve_linear_equation(image_maxRGB_reduced_01, illumination_map, method=solver, CG_prec=CG_prec, CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL)
    log_memory('bimef|solve_linear_equation|E')
    ######################################################

    ############ RESTORE REDUCED SIZE SMOOTH MATRIX TO FULL SIZE:  ###########################
    log_memory('bimef|imresize|B')
    image_maxRGB_01_smooth = imresize(image_maxRGB_reduced_01_smooth, size=image_maxRGB.shape)
    log_memory('bimef|imresize|E')
    ######################################################
    
    ############# CALCULATE WEIGHTS ###############################
    log_memory('bimef|weights|B')
    weights = np.power(image_maxRGB_01_smooth, enhance)  
    weights = np.expand_dims(weights, axis=2)
    weights  = np.where(weights>1,1,weights)
    log_memory('bimef|weights|E')
    ######################################################
    
    log_memory('bimef|autoscale_array|B')
    image_01 = autoscale_array(image)
    log_memory('bimef|autoscale_array|E')
    log_memory('bimef|np.zeros_like|B')
    dim_pixels = np.zeros_like(image_maxRGB_01_smooth)
    log_memory('bimef|np.zeros_like|E')
    
    if exposure_ratio==-1:
        log_memory('bimef|dim_threshold|B')
        dim_pixels = image_maxRGB_01_smooth<dim_threshold
        log_memory('bimef|dim_threshold|E')
        log_memory('bimef|get_dim_pixels|B')
        Y = get_dim_pixels(image_01, dim_pixels, dim_size=dim_size) 
        log_memory('bimef|get_dim_pixels|E')
        log_memory('bimef|optimize_exposure_ratio|B')
        exposure_ratio = optimize_exposure_ratio(Y, a, b, lo=lo, hi=hi, npoints=npoints)
        log_memory('bimef|optimize_exposure_ratio|E')
    
    log_memory('bimef|applyK|B')
    image = applyK(image_01, exposure_ratio, a, b, verbose=verbose) 
    log_memory('bimef|applyK|E')
    log_memory('bimef|np.where|B')
    image = np.where(image>1,1,image)    
    log_memory('bimef|np.where|E')
    ############ Final Result:  ###########################
    #enhanced_image =  image_01 * weights + image_exposure_adjusted * (1 - weights)   
    log_memory('bimef|result|B')
    image = image * (1 - weights)
    log_memory('bimef|result|1')
    image = image + image_01 * weights
    log_memory('bimef|result|2')
    image = (255 * image).astype(np.uint8)
    log_memory('bimef|result|E')
    ##################################################
    
    toc = datetime.now()

    if print_info:
        print(f'[{datetime.now().isoformat()}] exposure_ratio: {exposure_ratio:.4f}, enhance: {enhance:.4f}, lamda: {lamda:.4f}, scale: {scale:.4f}, runtime: {(toc-tic).total_seconds():.4f}s')

    log_memory('bimef||E')
    
    log_memory('bimef|gc.collect|B')
    gc.collect()
    log_memory('bimef|gc.collect|E')

    return image