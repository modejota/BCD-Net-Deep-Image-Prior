import os
import sys
import torch
import shutil
import numpy as np
from tkinter import Tk, filedialog


def rgb2od(rgb):
    """Converts RGB image to optical density (OD_RGB) image.
    Args:
        rgb: RGB image tensor / numpy array. Shape: (batch_size, 3, H, W)
    Returns:
        od: OD_RGB image tensor / numpy array. Shape: (batch_size, 3, H, W)
    """
    if isinstance(rgb, np.ndarray):
        od = -np.log((rgb + 1.0) / 256.0) / np.log(256.0)
    elif isinstance(rgb, torch.Tensor):
        od = -torch.log((rgb + 1.0) / 256.0) / np.log(256.0)
    return od

def od2rgb(od):
    """Converts optical density (OD) image to RGB image.
    Args:
        od: OD image tensor / numpy array. Shape: (batch_size, 3, H, W)
    Returns:
        rgb: RGB image tensor / numpy array. Shape: (batch_size, 3, H, W)
    """
    if isinstance(od, np.ndarray):
        rgb = 256.0 * np.exp(-od * np.log(256.0)) - 1.0
    elif isinstance(od, torch.Tensor):
        rgb = 256.0 * torch.exp(-od * np.log(256.0)) - 1.0
    return rgb

def direct_deconvolution(Y, M):
    """Solves Y = MC using least squares.

    Args:
        Y: OD image tensor / numpy array. Shape: (batch_size, 3, H, W)
        M: color matrix tensor / numpy array. Shape: (batch_size, 3, 2)
    """

    squeeze_at_end = False
    if len(Y.shape) == 3:
        Y = Y.unsqueeze(0)
        M = M.unsqueeze(0)
        squeeze_at_end = True

    batch_size, _, H, W = Y.shape

    if isinstance(Y, np.ndarray):
        Y = Y.reshape(batch_size, 3, -1) # (batch_size, 3, H*W)
        M = M.reshape(batch_size, 3, 2) # (batch_size, 3, 2)
        C = np.linalg.lstsq(M, Y, rcond=None)[0] # (batch_size, 2, H*W)
        C = C.reshape(batch_size, 2, H, W) # (batch_size, 2, H, W)
    elif isinstance(Y, torch.Tensor):
        Y = Y.view(batch_size, 3, -1) # (batch_size, 3, H*W)
        M = M.view(batch_size, 3, 2) # (batch_size, 3, 2)
        C = torch.linalg.lstsq(M, Y).solution # (batch_size, 2, H*W)
        C = C.view(batch_size, 2, H, W) # (batch_size, 2, H, W)

    if squeeze_at_end:
        C = C.squeeze(0)
    return C

def peak_signal_noise_ratio(A, B, max=255.0):
    """
    input:
        A: tensor of shape (N, C, H, W)
        B: tensor of shape (N, C, H, W)
    return:
        psnr: tensor of shape (N, )
    """
    if max is None:
        max = torch.max(A)
    mse = torch.mean((A - B)**2, dim=(1,2,3))
    psnr = 10 * torch.log10(max**2 / mse)
    return psnr

def structural_similarity(A, B):
    """
    Calculate Structural Similarity Index (SSIM) between two tensors.

    Parameters:
        A: tensor of shape (N, C, H, W)
        B: tensor of shape (N, C, H, W)

    Returns:
        ssim: tensor of shape (N, )
    """
    N, C, H, W = A.shape

    A = A.reshape(N, -1)  # Reshape to (N, C*H*W)
    B = B.reshape(N, -1)

    data_range = 255.0

    mu_A = torch.mean(A, dim=1)  # Compute mean along (C*H*W) -> (N, )
    mu_B = torch.mean(B, dim=1)
    var_A = torch.var(A, dim=1)  # Compute variance along (C*H*W) -> (N, )
    var_B = torch.var(B, dim=1)

    # Compute covariance of A and B
    cov_AB = torch.mean((A - mu_A.view(-1, 1)) * (B - mu_B.view(-1, 1)), dim=1)  # -> (N, )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Compute SSIM index for each N
    ssim = (2 * mu_A * mu_B + c1) * (2 * cov_AB + c2) / ((mu_A ** 2 + mu_B ** 2 + c1) * (var_A + var_B + c2))

    return ssim

def random_ruifrok_matrix_variation(std):
    """
    input:
        std: standard deviation of the gaussian noise
    return:
        Hgauss: new distribution for H channel
        Egauss: new distribution for E channel
    """
    M_Ruifrok=np.array([[0.644211  , 0.092789  , 0.63598099],
                        [0.716556  , 0.954111  , 0.        ],
                        [0.266844  , 0.283111  , 0.77170472]])
    M_Ruifrok=M_Ruifrok[:,:3]   # Only H&E

    Hgauss=np.abs(np.random.normal(loc=M_Ruifrok[:,0], scale=std)).reshape([3,1])
    Egauss=np.abs(np.random.normal(loc=M_Ruifrok[:,1], scale=std)).reshape([3,1])

    # Normalize to ensure norm=1 (abs is to avoid negative values)
    Hgauss /= np.linalg.norm(Hgauss)
    Egauss /= np.linalg.norm(Egauss)

    return Hgauss, Egauss

def askforimageviaGUI(initialdirectory="."):
    Tk().withdraw()
    try:
        img = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero", filetypes=[
            ("Image files", (".jpg", ".png", ".tif")),
        ])
    except(OSError, FileNotFoundError):
        print(f'No se ha podido abrir el fichero seleccionado.')
        sys.exit(100)
    except Exception as error:
        print(f'Ha ocurrido un error: <{error}>')
        sys.exit(101)
    if len(img) == 0 or img is None:
        print(f'No se ha seleccionado ningún archivo.')
        sys.exit(102)
    return img

def askforPyTorchWeightsviaGUI(initialdirectory="."):
    Tk().withdraw()
    try:
        weights = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero", filetypes=[
            ("PyTorch weights", (".pth", '.pt')),
        ])
    except(OSError, FileNotFoundError):
        print(f'No se ha podido abrir el fichero seleccionado.')
        sys.exit(100)
    except Exception as error:
        print(f'Ha ocurrido un error: <{error}>')
        sys.exit(101)
    if weights is None:
        print(f'No se ha seleccionado ningún archivo.')
        sys.exit(102)
    return weights

def askforCSVfileviaGUI(initialdirectory="."):
    Tk().withdraw()
    try:
        csv = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero", filetypes=[
            ("CSV files", (".csv")),
        ])
    except(OSError, FileNotFoundError):
        print(f'No se ha podido abrir el fichero seleccionado.')
        sys.exit(100)
    except Exception as error:
        print(f'Ha ocurrido un error: <{error}>')
        sys.exit(101)
    if csv is None:
        print(f'No se ha seleccionado ningún archivo.')
        sys.exit(102)
    return csv

def generate_reduced_dataset_from_full_dataset(
        directorio_origen = '/home/modejota/Deep_Var_BCD/results_full_datasets/',
        directorio_destino = '/home/modejota/Deep_Var_BCD/results_reduced_datasets/',
        archivos_a_copiar = ['Colon_0.csv', 'Colon_6.csv', 'Lung_0.csv', 'Lung_48.csv', 'Breast_0.csv', 'Breast_48.csv']
):
    """
    Copia archivos desde el directorio de origen al directorio de destino y elimina
    los archivos que no tengan nombres especificados en la lista de nombres permitidos.
    Esencialmente, genera un subconjunto de un conjunto de datos.
    
    :param directorio_origen: Ruta al directorio de origen.
    :param directorio_destino: Ruta al directorio de destino.
    :param archivos_a_copiar: Lista de nombres de archivos permitidos.
    """

    if not os.path.exists(directorio_origen):
        print(f'El directorio de origen <{directorio_origen}> no existe.')
        return

    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    for root, _, files in os.walk(directorio_origen):
        # Construye la ruta correspondiente en el directorio de destino
        ruta_destino = os.path.join(directorio_destino, os.path.relpath(root, directorio_origen))
        
        # Crea el directorio en el destino si no existe
        os.makedirs(ruta_destino, exist_ok=True)
        
        for file in files:
            ruta_origen = os.path.join(root, file)
            ruta_destino_archivo = os.path.join(ruta_destino, file)
            
            if file not in archivos_a_copiar:
                # Elimina el archivo si no está en la lista
                os.remove(ruta_origen)
            else:
                # Copia el archivo al directorio de destino
                shutil.copy2(ruta_origen, ruta_destino_archivo)
