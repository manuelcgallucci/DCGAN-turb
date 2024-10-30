from time import time
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Optional
from pathlib import Path
import torch

from train_environment import TrainingEnvironment, time_training
import utility as ut

class TrainingEnvironmentGeneratorWithStrucutres(TrainingEnvironment):
    def __init__(self, epochs: int, lr: float):
        super().__init__(epochs, lr)
        self.criterionMSE = torch.nn.MSELoss()
        for loss_name in [f"loss_mse_{x}" for x in ["total", "s2", "skewness", "flatness"]]:
            self.generator_data[loss_name] = np.zeros((epochs))
               
    def _generate_samples(self, batch_idx: int, batch_size: int, n_samples: int, length_samples: int, no_grad: bool): 
        structure_s2, structure_skewness, structure_flatness = self._retrieve_real_data_structure_functions(batch_idx, self.device)
        noise = torch.randn((batch_size, n_samples, length_samples), device=self.device)
            
        # noise is batch_size x 1 x length_samples
        # strucure_x is batch_size x 100
        # stack them to get a noise that is batch_size x 1 x lenght_samples + 300

        structures = torch.stack([structure_s2, structure_skewness, structure_flatness], dim=-1).view(batch_size, -1)  # batch_size x 300
        structures = structures.unsqueeze(1).expand(-1, n_samples, -1)  # batch_size x n_samples x 300

        combined_input = torch.cat((noise, structures), dim=-1)  # batch_size x 1 x (length_samples + 300)
        if no_grad:
            with torch.no_grad():
                return self.generator_data["model"](combined_input)[:,:,150:-149]
        return self.generator_data["model"](combined_input)[:,:,150:-149]
    
    @time_training
    def train(self, dataloader_train: DataLoader, structure_scales: List[int], weights: torch.Tensor, k_epochs_g: int, k_epochs_d: int, beta: float, checkpoint_epochs: Optional[int]=None, output_path: Optional[Path]=None, verbose:bool=True):
        if checkpoint_epochs is not None and output_path is not None and not output_path.exists():
            raise AttributeError("Trying to save periodically to a path that does not exist")

        for _, data in enumerate(dataloader_train):
            noise_shape = (1, data.shape[2])
            break
        
        self._cache_real_data_structure_functions(dataloader_train, structure_scales)
        self._cache_target_tensors(dataloader_train, self.device)

        for epoch in range(self.epochs):
            for batch_idx, real_data in enumerate(dataloader_train):
                real_data = real_data.to(self.device).float()
                batch_size = real_data.shape[0]

                for kd in range(k_epochs_d):
                    for discriminator_data in self.discriminators_data.values():
                        discriminator_data["optim"].zero_grad()
                    
                    targets = self._retrieve_target_tensors_ones_cache(batch_idx)
                    loss_samples, loss_s2, loss_skewness, loss_flatness = self._get_discriminator_loss_from_cache_data(real_data, batch_idx, targets, weights, no_grad=False)
                    loss_real = 0.5 * loss_samples + loss_s2 + loss_skewness + loss_flatness

                    self._save_discriminator_data(loss_samples, loss_s2, loss_skewness, loss_flatness, epoch, k_epochs_d, "real")

                    fake_data = self._generate_samples(batch_idx, batch_size, noise_shape[0], noise_shape[1], no_grad=True)										
                    targets = self._retrieve_target_tensors_zeros_cache(batch_idx)
                    loss_samples, loss_s2, loss_skewness, loss_flatness = self._get_discriminator_loss(fake_data, targets, structure_scales, weights, no_grad=False)
                    loss_fake = 0.5 * loss_samples + loss_s2 + loss_skewness + loss_flatness

                    self._save_discriminator_data(loss_samples, loss_s2, loss_skewness, loss_flatness, epoch, k_epochs_d, "fake")

                    loss_discriminators = (loss_real + loss_fake) / 2					
                    loss_discriminators.backward()

                    for discriminator_data in self.discriminators_data.values():
                        discriminator_data["optim"].step()

                    self._log_discriminator_data(epoch, loss_real, loss_fake, loss_discriminators, k_epochs_d)
                
                for kg in range(k_epochs_g):
                    self.generator_data["optim"].zero_grad()

                    generated_signal = self._generate_samples(batch_idx, batch_size, noise_shape[0], noise_shape[1], no_grad=False)
                    
                    targets = self._retrieve_target_tensors_ones_cache(batch_idx)
                    # loss_samples, loss_s2, loss_skewness, loss_flatness = self._get_discriminator_loss(generated_signal, targets, structure_scales, weights, no_grad=True)
                   
                    predictions = self.discriminators_data["samples"]["model"](generated_signal)
                    generated_structure_s2, generated_structure_skewness, generated_structure_flatness = ut.calculate_structure_noInplace_new(generated_signal, structure_scales, device=self.device)
                    loss_samples, loss_s2, loss_skewness, loss_flatness = self._calculate_discriminator_losses(predictions, targets, weights, generated_structure_s2, generated_structure_skewness, generated_structure_flatness, no_grad=False)
                
                    structure_s2, structure_skewness, structure_flatness = self._retrieve_real_data_structure_functions(batch_idx, self.device)

                    loss_mse_s2 = self.criterionMSE(generated_structure_s2, structure_s2)
                    loss_mse_skewness = self.criterionMSE(generated_structure_skewness, structure_skewness)
                    loss_mse_flatness = self.criterionMSE(generated_structure_flatness, structure_flatness)

                    loss_mse = loss_mse_s2 + loss_mse_flatness + loss_mse_skewness
                    loss_generator =  0.5 * loss_samples + loss_s2 + loss_skewness + loss_flatness + beta * loss_mse
                    loss_generator.backward()

                    self.generator_data["optim"].step()
                    
                    self._log_generator_data(epoch, loss_samples, loss_samples, loss_s2, loss_skewness, loss_flatness, loss_generator)			
                    self.generator_data["loss_mse_total"][epoch] += loss_mse.item() / k_epochs_g
                    self.generator_data["loss_mse_s2"][epoch] += loss_mse_s2.item() / k_epochs_g
                    self.generator_data["loss_mse_skewness"][epoch] += loss_mse_skewness.item() / k_epochs_g
                    self.generator_data["loss_mse_flatness"][epoch] += loss_mse_flatness.item() / k_epochs_g



            self._print_epoch_message(epoch, np.log10(self.generator_data["loss_total"][epoch]), np.log10(self.discriminator_history["loss_total"][epoch]), verbose)
            if verbose: 			
                print(f'\t\t Generator MSELlos : {np.log10(self.generator_data["loss_mse_total"][epoch]):7.4f} \t/\t\t s2 LLoss : {np.log10(self.generator_data["loss_mse_s2"][epoch]):7.4f} \t skewness LLoss : {np.log10(self.generator_data["loss_mse_skewness"][epoch]):7.4f} \t flatness LLoss : {np.log10(self.generator_data["loss_mse_flatness"][epoch]):7.4f}')

            self._save_models_checkpoint(epoch, checkpoint_epochs, output_path, verbose)
