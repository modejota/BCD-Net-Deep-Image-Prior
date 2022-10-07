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
    Y = Y.reshape(c, -1) # channels first
    ct = np.linalg.lstsq(M,Y,rcond=None)[0]
    ct = ct.reshape((ct.shape[0], n, m))
    return ct

def direct_deconvolution_torch(Y, M):
    (h,w,c) = Y.shape
    Y = Y.reshape(-1,c).T # channels first
    ct, _ = torch.lstsq(Y, M)
    ct = ct.reshape((ct.shape[0], h, w))
    return ct

def C_to_RGB_np(C, M):
    """
    C: (bs, ns, h, w)
    M: (bs, 3, ns)
    Returns: (bs, h, w, 3)
    """
    if len (C.shape) == 3:
        C = C.unsqueeze(0)
        M = M.unsqueeze(0)
    C_rgb = np.einsum('bcs, bswh -> bschw', M, C)
    C_rgb = np.clip(od2rgb_np(C_rgb), 0, 255).astype(np.uint8)
    C_rgb = C_rgb.transpose(0,1,3,4,2)
    if C.shape[0]==1:
        C_rgb = C_rgb.squeeze(0)
    return C_rgb

def C_to_RGB_torch(C, M):
    """
    C: (bs, ns, h, w)
    M: (bs, 3, ns)
    Returns: (bs, 3, h, w)
    """
    if len (C.shape) == 3:
        C = C.unsqueeze(0)
        M = M.unsqueeze(0)
    C_rgb = torch.einsum('bcs, bswh -> bschw', M, C)
    C_rgb = torch.clamp(od2rgb_torch(C_rgb), 0, 255)
    return C_rgb