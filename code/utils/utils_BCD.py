import torch
import numpy as np

def rgb2od_torch(rgb):
    od = (-torch.log((rgb + 1) / 256))
    return od

def rgb2od_np(rgb):
    od = (-np.log((rgb.astype(float) + 1) / 256))
    return od

def od2rgb_np(od):
    return 256.0*np.exp(-od)-1.0

def od2rgb_torch(od):
    return 256.0*torch.exp(-od)-1.0

def normalize_to1(data, max_val=np.log(256.0), min_val=0.0):
    return (data - min_val) / (max_val - min_val)

def undo_normalization(data, max_val=np.log(256.0), min_val=0.0):
    return data * (max_val - min_val) + min_val

def direct_deconvolution_np(Y, M):
    (n,m,c) = Y.shape
    Y = Y.transpose(2, 0, 1) # channels first
    Y = Y.reshape(c, -1)
    ct = np.linalg.lstsq(M,Y,rcond=None)[0]
    ct = ct.reshape((ct.shape[0], n, m))
    return ct

def direct_deconvolution_torch(Y, M):
    (h,w,c) = Y.shape
    Y = Y.tranpose(2, 0, 1) # channels first
    Y = Y.view(c, -1)
    ct, _ = torch.lstsq(Y, M)
    ct = ct.reshape((ct.shape[0], h, w))
    return ct

def C_to_OD_np(C, M):
    """
    C: (bs, ns, h, w)
    M: (bs, 3, ns)
    Returns: (bs, h, w, 3)
    """
    if len (C.shape) == 3:
        C = C.unsqueeze(0)
        M = M.unsqueeze(0)
    C_od = np.einsum('bcs, bshw -> bschw', M, C)
    #C_od = undo_normalization(C_od)
    C_od = C_od.transpose(0,1,3,4,2)
    if C.shape[0]==1:
        C_od = C_od.squeeze(0)
    return C_od

def C_to_OD_torch(C, M):
    """
    C: (bs, ns, h, w)
    M: (bs, 3, ns)
    Returns: (bs, 3, h, w)
    """
    if len (C.shape) == 3:
        C = C.unsqueeze(0)
        M = M.unsqueeze(0)
    C_od = torch.einsum('bcs, bshw -> bschw', M, C)
    #C_od = undo_normalization(C_od)
    return C_od

def C_to_RGB_np(C, M):
    """
    C: (bs, ns, h, w)
    M: (bs, 3, ns)
    Returns: (bs, h, w, 3)
    """
    C_od = C_to_OD_np(C, M)
    C_rgb = od2rgb_np(C_od)
    return C_rgb