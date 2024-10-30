import numpy as np
import torch
import random, string
import json
from pathlib import Path

# Tested at 50 tests, 32 batch size, 2**15 length, 100 scales
# 0.017 gpu
def calculate_s2(signal, scales, device="cpu"):
    if isinstance(signal, np.ndarray):
        signal = torch.Tensor(signal)
        signal = signal[:, None, :]
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''      
    s2 = torch.zeros((signal.shape[0],1,len(scales)), dtype=torch.float32, device=device)

    # We normalize the image by centering and standarizing it
    Nreal=signal.size()[0]
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        s2[:,:,idx] = torch.log(torch.mean(torch.square(tmp[:,:,scale:]-tmp[:,:,:-scale]), dim=2))
        
    if device != "cpu":
        s2 = s2.detach().cpu()
    return s2[:,0,:].numpy()


def calculate_structure(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''      
    # This contains in order, log(s2), skewness, log(flatness / 3)
    structure_f = torch.zeros((signal.shape[0],3,len(scales)), dtype=torch.float32, device=device)

    # We normalize the image by centering and standarizing it
    Nreal=signal.size()[0]
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        incrs = tmp[:,0,scale:]-tmp[:,0,:-scale]
        structure_f[:,0,idx] = torch.mean(torch.square(incrs), dim=1)
        
        stdincrs = torch.std(incrs, dim=1)
        incrsnorm = (incrs - torch.nanmean(incrs, dim=1)[:,None]) / stdincrs[:,None] # Batch x Length
        
        structure_f[:,1,idx] = torch.nanmean(torch.pow(incrsnorm,3), dim=1)
        structure_f[:,2,idx] = torch.nanmean(torch.pow(incrsnorm, 4), dim=1)

    # log s2 and log flatness / 3
    structure_f[:,0,:] = torch.log(structure_f[:,0,:])
    structure_f[:,2,:] = torch.log(structure_f[:,2,:] / 3)

    return structure_f

def calculate_structure_noInplace(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''      
    # This contains in order, log(s2), skewness, log(flatness / 3)
    structure_f = torch.zeros((signal.shape[0],3,len(scales)), dtype=torch.float32, device=device)

    # We normalize the image by centering and standarizing it
    Nreal=signal.size()[0]
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        incrs = tmp[:,0,scale:]-tmp[:,0,:-scale]
        structure_f[:,0,idx] = torch.log(torch.mean(torch.square(incrs), dim=1))
        
        stdincrs = torch.std(incrs, dim=1)
        incrsnorm = (incrs - torch.nanmean(incrs, dim=1)[:,None]) / stdincrs[:,None] # Batch x Length
        
        structure_f[:,1,idx] = torch.nanmean(torch.pow(incrsnorm,3), dim=1)
        structure_f[:,2,idx] = torch.log(torch.nanmean(torch.pow(incrsnorm, 4), dim=1) / 3)

    # log s2 and log flatness / 3
    #log_s2 = torch.log(structure_f[:,0,:])
    #structure_f[:,0,:] = log_s2
    
    #log_flatness = torch.log(structure_f[:,2,:] / 3)
    #structure_f[:,2,:] = log_flatness
    
    return structure_f
    
# Device has to be cpu since histogram is not yet implemented on CUDA
def calculate_histogram(signal, scales, n_bins, device="cpu", normalize_incrs=True):
    if device != "cpu":
        return 
    Nreal=signal.size()[0]

    histograms = np.zeros((Nreal, n_bins))
    bins = np.zeros((Nreal, n_bins+1))
    # We normalize the image by centering and standarizing it
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        incrs = tmp[:,0,scale:]-tmp[:,0,:-scale] # Incrs is Nbatch x L 

        if normalize_incrs:
            incrs= torch.div( (incrs - torch.mean(incrs,axis=1)[:,None] ), torch.std(incrs,axis=0))


        histograms[idx, :], bins[idx, :] = torch.histogram(incrs, n_bins, density=True)

    return histograms, bins

def calculate_structure_noInplace_new(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''      
    # This contains in order, log(s2), skewness, log(flatness / 3)
    structure_s2 = torch.zeros((signal.shape[0],len(scales)), dtype=torch.float32, device=device)
    strucutre_skewness = torch.zeros((signal.shape[0],len(scales)), dtype=torch.float32, device=device)
    strucutre_flatness = torch.zeros((signal.shape[0],len(scales)), dtype=torch.float32, device=device)

    # We normalize the image by centering and standarizing it
    Nreal=signal.size()[0]
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        incrs = tmp[:,0,scale:]-tmp[:,0,:-scale]
        structure_s2[:,idx] = torch.log(torch.mean(torch.square(incrs), dim=1))
        
        stdincrs = torch.std(incrs, dim=1)
        incrsnorm = (incrs - torch.nanmean(incrs, dim=1)[:,None]) / stdincrs[:,None] # Batch x Length
        
        strucutre_skewness[:,idx] = torch.nanmean(torch.pow(incrsnorm,3), dim=1)
        strucutre_flatness[:,idx] = torch.log(torch.nanmean(torch.pow(incrsnorm, 4), dim=1) / 3)

    return structure_s2, strucutre_skewness, strucutre_flatness


def get_structure_scales(length: int=100):
    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    return scales[:length]

def get_uuid(length=6):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))

def save_meta(meta_file_path: Path, meta_dict: dict):
    with open(str(meta_file_path), "w") as f:
        json.dump(meta_dict, f, indent=1)
