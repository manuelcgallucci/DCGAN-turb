import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image
import glob
import re 
import os 

import utility as ut

import torch
from model_generator import CNNGeneratorBigConcat as CNNGenerator


data_dir = "./generated/model/"
temp_dir = "./temp/"
save = True
display = False

def main():
	samples_len = 2**15
	print("Plotting histogram and structure functions with an output length of:", samples_len)
    
	# plot_real_data()
	# plot_history_structureTraining()
	# plot_compare_structure(eta=5, L=2350, len_=samples_len)
	plot_histogram(n_samples=64, len_=samples_len, scales=[2,4,8,16,128,256,1024,4096,8192,16384])

def plot_history_structureTraining():

	history = np.load(data_dir + 'metaEvo.npz')

	plt.figure()
	plt.plot(history["loss_fake"])
	plt.plot(history["loss_fake_s2"])
	plt.plot(history["loss_fake_skewness"])
	plt.plot(history["loss_fake_flatness"])
	plt.plot(history["loss_fake_total"])
	plt.title("Losses of Discriminator fake samples")
	plt.legend(["Samples", "S2", "skewnnes", "Flatness", "Total"])
	if save: plt.savefig(data_dir + "loss_discriminator_fake.png")
	if display: plt.show()

	plt.figure()
	plt.plot(history["loss_real"])
	plt.plot(history["loss_real_s2"])
	plt.plot(history["loss_real_skewness"])
	plt.plot(history["loss_real_flatness"])
	plt.plot(history["loss_real_total"])
	plt.title("Losses of Discriminator real samples")
	plt.legend(["Samples", "S2", "skewnnes", "Flatness", "Total"])
	if save: plt.savefig(data_dir + "loss_discriminator_real.png")
	if display: plt.show()

	plt.figure()
	plt.plot(history["loss_real_s2"])
	plt.plot(history["loss_real_skewness"])
	plt.plot(history["loss_real_flatness"])
	plt.plot(history["loss_fake_s2"])
	plt.plot(history["loss_fake_skewness"])
	plt.plot(history["loss_fake_flatness"])
	plt.title("Losses of Discriminator in structure functions")
	plt.legend(["real_S2", "real_skewnnes", "real_Flatness", "fake_S2", "fake_skewnnes", "fake_Flatness"])
	if save: plt.savefig(data_dir + "loss_discriminator_structures.png")
	if display: plt.show()

	plt.figure()
	plt.plot(history["loss_g"])
	plt.plot(history["loss_g_s2"])
	plt.plot(history["loss_g_skewness"])
	plt.plot(history["loss_g_flatness"])
	plt.plot(history["loss_g_total"])
	plt.title("Losses of Generator samples")
	plt.legend(["Samples", "S2", "skewnnes", "Flatness", "Total"])
	if save: plt.savefig(data_dir + "loss_generator.png")
	if display: plt.show()

	plt.figure()
	plt.plot(history["loss_real_total"])
	plt.plot(history["loss_fake_total"])
	plt.plot(history["loss_discriminator"])
	plt.title("Losses of Discriminator")
	plt.legend(["Real", "Fake", "Total"])
	if save: plt.savefig(data_dir + "loss_discriminator.png")
	if display: plt.show()

	plt.figure()
	plt.plot(history["loss_generator"])
	plt.plot(history["loss_discriminator"])
	plt.title("Losses of the models")
	plt.legend(["Generator", "Discriminator"])
	if save: plt.savefig(data_dir + "losses.png")
	if display: plt.show()

def plot_structure(n_samples=64, len_=2**15, edge=4096, device="cuda"):
    color=np.array([166, 178, 255])/255.0

    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]

    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(data_dir + 'generator.pt'))
    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
    
    generated_samples = generated_samples[:,:,edge:-edge]
    plt.figure()
    for k in range(n_samples):
        plt.plot(generated_samples[k,0,:].cpu(), linewidth=1.0)
    plt.title("Generated samples")
    if save: plt.savefig(data_dir + "s2_gen_samples.png")
    if display: plt.show()
    
    s2 = ut.calculate_s2(generated_samples, scales, device=device).cpu()
    log_scale = np.log(scales)
    plt.figure()
    for k in range(n_samples):
        plt.plot(log_scale, s2[k,0,:], color=color, linewidth=1.0)
    plt.plot(log_scale, torch.mean(s2[:,0,:], dim=0), 'r', linewidth=2.0)
    plt.title("Structure function on the samples")
    plt.xlabel("scales (log)")
    plt.xticks([x for x in range(0, int(np.ceil(log_scale[-1])))])
    plt.grid()
    if save: plt.savefig(data_dir + "s2_samples.png")
    if display: plt.show()

def plot_compare_structure(n_samples=64, len_=2**15, edge=4096, eta=None, L=None, device="cuda"):
    data_train = np.load('./data/data.npy')

    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]

    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(data_dir + 'generator.pt'))
    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
    generated_samples = generated_samples[:,:,edge:-edge]
    
    # for i in range(n_samples):
	#     generated_samples[i,0,:] = np.flip(generated_samples[i,0,:])
    generated_samples = generated_samples[:,0,:]
    
    generated_samples = np.flip(np.array(generated_samples.cpu()), axis=1).copy()
    generated_samples = generated_samples[:,None,:]
    generated_samples = torch.Tensor(generated_samples).to(device)
    # generated_samples = torch.Tensor(np.load(data_dir + 'samples.npz')["arr_0"])
    log_scale = np.log(scales)
    
    struct = ut.calculate_structure(generated_samples, scales, device=device)
    struct_mean_generated = torch.mean(struct[:,:,:], dim=0).cpu()
    struct_std_generate = torch.std(struct[:,:,:], dim=0).cpu()

    # TODO correct the data trin so this is not needed anymore
    data_train = np.flip(data_train, axis=1).copy()
    struct = ut.calculate_structure(torch.Tensor(data_train[:,None,:]), scales, device=device)
    struct_mean_real = torch.mean(struct[:,:,:], dim=0).cpu()
    struct_std_real = torch.std(struct[:,:,:], dim=0).cpu()

    names = ["s2", "skewness", "flatness"]
    ylabel = ["log(s2)", "skewness", "log(F / 3) - verify the 3"]
    vlines_lim = [(-7, 1), (-0.7, 0.1), (-0.2, 1)]
    for idx in range(3):
        plt.figure()
        plt.errorbar(log_scale, struct_mean_generated[idx,:], struct_std_generate[idx,:], color="red",  linewidth=2.0)
        plt.errorbar(log_scale, struct_mean_real[idx,:], struct_std_real[idx,:], linewidth=2.0, alpha=0.5)
        plt.title("Structure function {:s} on the samples".format(names[idx]))
        plt.xlabel("scales (log)")
        plt.ylabel(ylabel[idx])
        plt.xticks([x for x in range(0, int(np.ceil(log_scale[-1])))])
        plt.legend(["Generated", "Real"])
        plt.vlines(np.log(L), vlines_lim[idx][0], vlines_lim[idx][1], color='k', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.vlines(np.log(eta), vlines_lim[idx][0], vlines_lim[idx][1], color='k', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.grid()
        if save: plt.savefig(data_dir + "comparison_{:s}.png".format(names[idx]))
        if display: plt.show()

    # Compute and save structure function metrics
    mse_structure = torch.mean(torch.square(struct_mean_generated - struct_mean_real), dim=1)
    print(mse_structure)
    
def plot_histogram(n_samples=64, len_=2**15, edge=4096, n_bins=150, scales=[2**x for x in range(1,14)], device="cuda"):
    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(data_dir + 'generator.pt'))
    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
    
    generated_samples = generated_samples[:,:,edge:-edge].cpu()
    
    # generated_samples = torch.Tensor(np.load(data_dir + 'samples.npz')["arr_0"])
    # Histogram for the real samples
    #generated_samples = torch.Tensor(np.load('./data/data.npy')[:,None,:])

    # generated_samples = torch.Tensor(np.reshape( np.load('./data/full_signal.npy'), (1,-1))[:,None,:])
    # print(generated_samples.size())
    histograms, bins = ut.calculate_histogram(generated_samples, scales, n_bins, device="cpu", normalize_incrs=True) 
    histograms = np.log(histograms)

    histgn, binsg = np.histogram(np.random.randn(n_samples*len_), bins=n_bins, density=True)
    # tuple to set the limits of the scales and color
    params_ = [(32, "Dissipative", "black"), (512, "Inertial", "blue"), (32768, "Dissipative", "red")]
    delta_hist = 2
    plt.figure(figsize=(6,6))
    for idx, scale in enumerate(scales):
        param_idx = np.argmax(np.array([scale <= param[0] for param in params_]) > 0)
        plt.plot(bins[idx, 0:-1], delta_hist * (-idx) + histograms[idx,:], color=params_[param_idx][2], linewidth=1.5, label="sc:"+str(scale))
    plt.plot(binsg[0:-1],delta_hist * (-(len(scales)-1)) + np.log(histgn), "--", color="k", linewidth=1.5, alpha=0.8, label="Gaussian")
    plt.title("Histogram")
    plt.legend(bbox_to_anchor = (1.25, 0.6), loc='center right')
    plt.yticks([])
    plt.ylabel(r'$\log(P(\delta^{\prime}_l u(x)))$')
    plt.xlabel(r'$\delta^{\prime}_l u(x)$')
    plt.grid()
    plt.tight_layout()
    if save: plt.savefig(data_dir + "histogram.png")
    if display: plt.show()

def plot_real_data(device="cuda"):
    data_train = np.load('./data/data.npy')
    n_samples = data_train.shape[0]

    fig, ax = plt.subplots(8,8)
    fig.suptitle('Real samples')
    for k, axs in enumerate(fig.axes):
        axs.plot(data_train[k, :])
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])
    if save: plt.savefig("./data/samples_real.png")
    if display: plt.show()

    plt.figure()
    max_k = 64
    for k in range(data_train.shape[0]):
        if k >= max_k: break
        plt.plot(data_train[k, :], linewidth=1.0)
    fig.suptitle(str(max_k) + ' Real samples')
    if save: plt.savefig("./data/sample_once_real.png")
    if display: plt.show()

    color=np.array([166, 178, 255])/255.0

    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]
    
    s2 = ut.calculate_s2(torch.Tensor(data_train[:,None,:]), scales, device=device).cpu()
    log_scale = np.log(scales)
    plt.figure()
    for k in range(n_samples):
        plt.plot(log_scale, s2[k,0,:], color=color, linewidth=1.0)
    plt.plot(log_scale, torch.mean(s2[:,0,:], dim=0), 'r', linewidth=2.0)
    plt.title("Structure function on the real samples")
    plt.xlabel("scales (log)")
    plt.xticks([x for x in range(0, int(np.ceil(log_scale[-1])))])
    plt.grid()
    if save: plt.savefig("./data/s2_real.png")
    if display: plt.show()

if __name__ == '__main__':
    main()
