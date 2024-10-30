from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from model_generator import CNNGeneratorBigConcat as CNNGenerator
import utility as ut
import config

def load_data(data_path: Path, n_samples: int, sample_length: int) -> np.ndarray:
    data = np.load(data_path)
    loaded_data = np.zeros((n_samples, sample_length))
    for n in range(n_samples):
        loaded_data[n, :] = data[n * sample_length: (n + 1) * sample_length]
    return loaded_data

def generate_samples(generator, n_samples: int, sample_length: int, device: str) -> np.ndarray:
    noise = torch.randn((n_samples, 1, sample_length), device=device)
    with torch.no_grad():
        return generator(noise).detach().cpu().numpy()[:, 0, :]
    
def generate_samples_with_s(generator, s_functions, n_samples: int, sample_length: int, device: str) -> np.ndarray:
    noise = torch.randn((n_samples, 1, sample_length), device=device)
    
    print(noise.shape)
    print(s_functions[:,0,:][:,None,:].shape)
    noise = torch.cat((noise, s_functions[:,0,:][:,None,:], s_functions[:,1,:][:,None,:], s_functions[:,2,:][:,None,:]), dim=2)
    with torch.no_grad():
        return generator(noise).detach().cpu().numpy()[:, 0, :]

def plot_samples(samples: np.ndarray, output_file_dir: Path):
    """
        samples: size (n_samples, length)
    """
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(samples.T[:512, :], linewidth=1.0, alpha=0.7)
    plt.title("Generated samples")
    plt.savefig(output_file_dir)
    plt.close()

def plot_s2(s2: np.ndarray, log_scale: np.ndarray, output_file_dir: Path):
    """
        s2: size (n_samples, length_scales)
        log_scale: size (length_scales,)
    """
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(log_scale, s2.T, color="r", linewidth=1.0)
    plt.plot(log_scale, np.mean(s2, axis=0), 'k', linewidth=2.0)
    plt.title("Structure function on the samples")
    plt.xlabel("scales (log)")
    plt.xticks([x for x in range(0, int(np.ceil(log_scale[-1])))])
    plt.grid()
    plt.savefig(output_file_dir)
    plt.close()

def plot_history(history_dir: Path, output_dir: Path):
    history = np.load(history_dir)

    # plt.figure()
    # plt.plot(history["loss_fake"])
    # plt.plot(history["loss_fake_s2"])
    # plt.plot(history["loss_fake_skewness"])
    # plt.plot(history["loss_fake_flatness"])
    # plt.plot(history["loss_fake_total"])
    # plt.title("Losses of Discriminator fake samples")
    # plt.legend(["Samples", "S2", "skewnnes", "Flatness", "Total"])
    # if save: plt.savefig(data_dir + "loss_discriminator_fake.png")
    # if display: plt.show()

    # plt.figure()
    # plt.plot(history["loss_real"])
    # plt.plot(history["loss_real_s2"])
    # plt.plot(history["loss_real_skewness"])
    # plt.plot(history["loss_real_flatness"])
    # plt.plot(history["loss_real_total"])
    # plt.title("Losses of Discriminator real samples")
    # plt.legend(["Samples", "S2", "skewnnes", "Flatness", "Total"])
    # if save: plt.savefig(data_dir + "loss_discriminator_real.png")
    # if display: plt.show()

    # plt.figure()
    # plt.plot(history["loss_d_real_s2"])
    # plt.plot(history["loss_d_real_skewness"])
    # plt.plot(history["loss_d_real_flatness"])
    # plt.plot(history["loss_d_fake_s2"])
    # plt.plot(history["loss_d_fake_skewness"])
    # plt.plot(history["loss_d_fake_flatness"])
    # plt.title("Losses of Discriminator in structure functions")
    # plt.legend(["real_S2", "real_skewnnes", "real_Flatness", "fake_S2", "fake_skewnnes", "fake_Flatness"])
    # if save: plt.savefig(data_dir + "loss_discriminator_structures.png")
    # if display: plt.show()

    plt.figure()
    plt.plot(history["loss_g_samples"], "Samples")
    plt.plot(history["loss_g_s2"], "s2")
    plt.plot(history["loss_g_skewness"], "skewness")
    plt.plot(history["loss_g_flatness"], "flatness")
    plt.plot(history["loss_g_total"], "total")
    plt.title("Losses of Generator samples")
    plt.legend()
    plt.close()
    plt.savefig(output_dir / "loss_generator.png")
    
    plt.figure()
    plt.plot(np.log10(history["loss_generator"]), label="Generator")
    plt.plot(np.log10(history["loss_discriminator"]), label="Discriminator")
    plt.title("Losses of the models")
    plt.legend()
    plt.savefig(output_dir / "losses.png")
    plt.close()
    

def main(data_path: Path, model_uuid: str, output_path: Path, epoch: Optional[int], device: str):
    input_path = config.OUTPUT_PATH / model_uuid
    
    if epoch is not None:
        input_path = input_path / f"e_{epoch}"
        output_path = output_path / f"e_{epoch}"
        output_path.mkdir(exist_ok=True)
        if not input_path.exists():
            raise FileExistsError(f"File does not exist: {input_path}")
    else:
        # plot_history(input_path / config.EVOLUTION_FILE, output_path)
        pass

    generator_file_path = input_path / config.GENERATOR_FILE
    generator = CNNGenerator().to(device)
    generator.eval()
    generator.load_state_dict(torch.load(generator_file_path))

    # generated_samples = generate_samples(generator, 64, 2**15 + 300, device)

    scales = ut.get_structure_scales()
    data = load_data(data_path, 4, 2**15)
    # s2 = ut.calculate_s2(data, scales, device=device)
    # plot_s2(s2, np.log(scales), output_path / "s2_real.png")
    
    s_functions = ut.calculate_structure_noInplace(torch.Tensor(data)[:, None, :], scales, device=device)
    generated_samples = generate_samples_with_s(generator, s_functions, 4, 2**15, device)
    plot_samples(generated_samples[:4, :], output_path / "samples.png")

    # s2 = ut.calculate_s2(generated_samples, scales, device=device)
    # plot_s2(s2, np.log(scales), output_path / "s2.png")




if __name__ == "__main__":
    epoch = None
    model_uuid = "hX7r64"
    
    data_path = Path("data", "full_signal.npy")
    output_path = Path("output", model_uuid)
    output_path.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(data_path, model_uuid, output_path, epoch, device)