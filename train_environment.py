import torch 
from torch.utils.data import DataLoader
import numpy as np
from time import time
from functools import wraps

from typing import Tuple
from pathlib import Path
import config

import sys
from typing import List, Optional
from itertools import islice

# import dataloader as dl
import nn_definitions as nn_d
import utility as ut

from model_generator import CNNGeneratorBigConcat as CNNGenerator
from model_discriminator import DiscriminatorMultiNet16_4 as Discriminator
from model_discriminator import DiscriminatorStructures_v2 as DiscriminatorStructures


def time_training(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		verbose = kwargs.get("verbose", True)  # Check if verbose is enabled
		start_time = time()

		func(*args, **kwargs)

		end_time = time()
		elapsed_time = end_time - start_time
		if verbose:
			print(f"Training completed in {elapsed_time:.2f} seconds.")

		return elapsed_time		
	return wrapper

class TrainingEnvironment:
	def __init__(self, epochs: int, lr: float):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.lr = lr
		self.epochs = epochs

		assert self.device != "cpu", "Training cannot be run on the cpu"

		self.discriminator_history = {
			"loss_real_total": np.zeros((epochs)),
			"loss_fake_total": np.zeros((epochs)),
			"loss_total": np.zeros((epochs))
		}
		self.discriminators_data = {}
		self.discriminators_data["samples"] = self._initialize_model_data(Discriminator(), lr, epochs)
		self.discriminators_data["s2"] = self._initialize_model_data(DiscriminatorStructures(), lr, epochs)
		self.discriminators_data["skewness"] = self._initialize_model_data(DiscriminatorStructures(), lr, epochs)
		self.discriminators_data["flatness"] = self._initialize_model_data(DiscriminatorStructures(), lr, epochs)
		self.generator_data = self._initialize_model_data(CNNGenerator(), lr, epochs)
		for loss_name in [f"loss_{x}" for x in ["total", "samples", "s2", "skewness", "flatness"]]:
			self.generator_data[loss_name] = np.zeros((epochs))
		
		self.criterion = torch.nn.BCELoss().to(self.device)

	def _initialize_model_data(self, model, lr: float, epochs: int) -> dict:
		model_data = {}
		model_data["model"] = model.to(self.device).apply(nn_d.weights_init)
		model_data["optim"] = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
		model_data["optim"].zero_grad()
		model_data["loss_real"] = np.zeros((epochs))
		model_data["loss_fake"] = np.zeros((epochs))
		return model_data

	def _calculate_samples_loss(self, criterion, predictions, target, weights, device):
		loss = torch.zeros((1), device=device)
		for k in range(len(weights)):
			loss += weights[k] * criterion(predictions[:,k], target)
		return loss

	def _calculate_discriminator_losses(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, structure_s2: torch.Tensor, structure_skewness: torch.Tensor, structure_flatness: torch.Tensor, no_grad: bool):
		if no_grad:
			with torch.no_grad():
				loss_samples = self._calculate_samples_loss(self.criterion, predictions, targets, weights, self.device)
				loss_s2 = self.criterion(self.discriminators_data["s2"]["model"](structure_s2)[:, 0], targets)
				loss_skewness = self.criterion(self.discriminators_data["skewness"]["model"](structure_skewness)[:, 0], targets)
				loss_flatness = self.criterion(self.discriminators_data["flatness"]["model"](structure_flatness)[:, 0], targets)
		else:
			loss_samples = self._calculate_samples_loss(self.criterion, predictions, targets, weights, self.device)
			loss_s2 = self.criterion(self.discriminators_data["s2"]["model"](structure_s2)[:, 0], targets)
			loss_skewness = self.criterion(self.discriminators_data["skewness"]["model"](structure_skewness)[:, 0], targets)
			loss_flatness = self.criterion(self.discriminators_data["flatness"]["model"](structure_flatness)[:, 0], targets)
		return loss_samples, loss_s2, loss_skewness, loss_flatness

	def _get_discriminator_loss_from_cache_data(self, data: torch.Tensor,  batch_idx: int, targets: torch.Tensor, weights: torch.Tensor, no_grad: bool):
		predictions = self.discriminators_data["samples"]["model"](data)
		structure_s2, structure_skewness, structure_flatness = self._retrieve_real_data_structure_functions(batch_idx, self.device)

		return self._calculate_discriminator_losses(predictions, targets, weights, structure_s2, structure_skewness, structure_flatness, no_grad)
	
	def _get_discriminator_loss(self, data, targets, structure_scales: List[int], weights: torch.Tensor, no_grad: bool):
		predictions = self.discriminators_data["samples"]["model"](data)
		structure_s2, structure_skewness, structure_flatness = ut.calculate_structure_noInplace_new(data, structure_scales, device=self.device)
		
		return self._calculate_discriminator_losses(predictions, targets, weights, structure_s2, structure_skewness, structure_flatness, no_grad)
		
	def _generate_samples(self, batch_idx: int, batch_size: int, n_samples: int, length_samples: int, no_grad: bool): 
		noise = torch.randn((batch_size, n_samples, length_samples), device=self.device)
		if no_grad:
			with torch.no_grad():
				return self.generator_data["model"](noise)
		return self.generator_data["model"](noise)

	def _save_discriminator_data(self, loss_samples, loss_s2, loss_skewness, loss_flatness, epoch: int, k_epochs_d: int, data_type: str):
		data_type_key = f"loss_{data_type}"		
		self.discriminators_data["samples"][data_type_key][epoch] += loss_samples.item() / k_epochs_d
		self.discriminators_data["s2"][data_type_key][epoch] += loss_s2.item() / k_epochs_d
		self.discriminators_data["skewness"][data_type_key][epoch] += loss_skewness.item() / k_epochs_d
		self.discriminators_data["flatness"][data_type_key][epoch] += loss_flatness.item() / k_epochs_d
	
	def _cache_real_data_structure_functions(self, dataloader_train: DataLoader, structure_scales: List[int]):
		self.cache_structure_functions = [] 
		for batch_idx, real_data in enumerate(dataloader_train):
			self.cache_structure_functions.append(ut.calculate_structure_noInplace_new(real_data, structure_scales, device="cpu"))

	def _retrieve_real_data_structure_functions(self, batch_idx: int, device: str) -> Tuple:
		structure_s2, structure_skewness, structure_flatness = self.cache_structure_functions[batch_idx]
		return structure_s2.to(device), structure_skewness.to(device), structure_flatness.to(device)

	def _cache_target_tensors(self, dataloader: DataLoader, device: str):
		batch_size = dataloader.batch_size
		final_batch_size = len(next(islice(dataloader, len(dataloader) - 1, None)))

		self.cache_targets_last_batch_idx = len(dataloader) - 1
		self.cache_target_ones = [
			torch.ones((batch_size), device=device),
			torch.ones((final_batch_size), device=device)
		]
		self.cache_target_zeros = [
			torch.zeros((batch_size), device=device),
			torch.zeros((final_batch_size), device=device)
		]

	def _retrieve_target_tensors_ones_cache(self, batch_idx: int) -> torch.Tensor:
		return self.cache_target_ones[batch_idx == self.cache_targets_last_batch_idx]

	def _retrieve_target_tensors_zeros_cache(self, batch_idx: int) -> torch.Tensor:
		return self.cache_target_ones[batch_idx == self.cache_targets_last_batch_idx]

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
					loss_samples, loss_s2, loss_skewness, loss_flatness = self._get_discriminator_loss(fake_data, targets, structure_scales, weights)
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
					loss_samples, loss_s2, loss_skewness, loss_flatness = self._get_discriminator_loss(generated_signal, targets, structure_scales, weights, no_grad=True)

					loss_generator = beta * ( 0.5 * loss_samples + loss_s2 + loss_skewness + loss_flatness)
					loss_generator.backward()

					self.generator_data["optim"].step()
					self._log_generator_data(epoch, loss_samples, loss_samples, loss_s2, loss_skewness, loss_flatness, loss_generator)			

			self._print_epoch_message(epoch, np.log10(self.generator_data["loss_total"][epoch]), np.log10(self.discriminator_history["loss_total"][epoch]), verbose)
			self._save_models_checkpoint(epoch, checkpoint_epochs, output_path, verbose)

	def _log_discriminator_data(self, epoch: int, loss_real: torch.Tensor, loss_fake: torch.Tensor, loss_discriminators: torch.Tensor, k_epochs_d: int):
			self.discriminator_history["loss_real_total"][epoch] += loss_real.item() / k_epochs_d
			self.discriminator_history["loss_fake_total"][epoch] += loss_fake.item() / k_epochs_d
			self.discriminator_history["loss_total"][epoch] += loss_discriminators.item() / k_epochs_d
				

	def _log_generator_data(self, epoch: int, loss_samples: torch.Tensor, loss_s2: torch.Tensor, loss_skewness: torch.Tensor, loss_flatness: torch.Tensor, loss_generator: torch.Tensor, k_epochs_g:int):
		self.generator_data["loss_samples"][epoch] += loss_samples.item() / k_epochs_g
		self.generator_data["loss_s2"][epoch] += loss_s2.item() / k_epochs_g
		self.generator_data["loss_skewness"][epoch] += loss_skewness.item() / k_epochs_g
		self.generator_data["loss_flatness"][epoch] += loss_flatness.item() / k_epochs_g
		self.generator_data["loss_total"][epoch] += loss_generator.item() / k_epochs_g	

	def _save_models_checkpoint(self, epoch: int, checkpoint_epochs: Optional[int], output_path: Optional[Path], verbose: bool):
		if checkpoint_epochs is not None:
			if (epoch + 1) % checkpoint_epochs == 0:
				if output_path is not None:
					out = output_path / f"e_{epoch+1}"
					out.mkdir(exist_ok=True)
					try:
						self.save_models(out)
						if verbose: print(f"Successfully saved checkpoint model in {out}")
					except Exception as e:
						print(f"Failed to save checkpoint model in {out}. Reason: {e}")
				else:
					print(f"Failed to save checkpoint model, maybe output path is not correctly defined? Output path: {out}")

	def _print_epoch_message(self, epoch: int, generator_llos: float, discriminator_llos: float, verbose: bool):
		if verbose:
			print(f'Epoch [{epoch+1}/{self.epochs}] -\t Generator LLoss : {generator_llos:7.4f} \t/\t\t Discriminator LLoss : {discriminator_llos:7.4f}')
			sys.stdout.flush()
		
	def save_metadata(self, out_dir: Path):
		np.savez(out_dir / config.EVOLUTION_FILE, \
			loss_d_fake_samples = self.discriminators_data["samples"]["loss_fake"], \
			loss_d_fake_s2 = self.discriminators_data["s2"]["loss_fake"], \
			loss_d_fake_skewness = self.discriminators_data["skewness"]["loss_fake"], \
			loss_d_fake_flatness = self.discriminators_data["flatness"]["loss_fake"], \
			loss_d_fake_total = self.discriminator_history["loss_fake_total"], \
			
			loss_d_real_samples = self.discriminators_data["samples"]["loss_fake"], \
			loss_d_real_s2 = self.discriminators_data["s2"]["loss_fake"], \
			loss_d_real_skewness = self.discriminators_data["skewness"]["loss_fake"], \
			loss_d_real_flatness = self.discriminators_data["flatness"]["loss_fake"], \
			loss_d_real_total = self.discriminator_history["loss_real_total"], \
			
			loss_g_samples = self.generator_data["loss_samples"], \
			loss_g_s2 = self.generator_data["loss_s2"], \
			loss_g_skewness = self.generator_data["loss_skewness"], \
			loss_g_flatness = self.generator_data["loss_flatness"], \
			
			loss_discriminator = self.discriminator_history["loss_total"], \
			loss_generator = self.generator_data["loss_total"]
		)

	def save_models(self, output_path: Path):
		torch.save(self.generator_data["model"].state_dict(), output_path / config.GENERATOR_FILE)
		for discriminator_name, file_name in zip(["samples", "s2", "skewness", "flatness"], [config.DISCRIMINATOR_SCALES_FILE, config.DISCRIMINATOR_S2_FILE, config.DISCRIMINATOR_SKEWNESS_FILE, config.DISCRIMINATOR_FLATNESS_FILE]):
			torch.save(self.discriminators_data[discriminator_name]["model"].state_dict(), output_path / file_name)
	
	def load_from_checkpoint(self, models_path: Path):
		self.generator_data["model"].load_state_dict(torch.load(models_path / config.GENERATOR_FILE))
		for discriminator_name, file_name in zip(["samples", "s2", "skewness", "flatness"], [config.DISCRIMINATOR_SCALES_FILE, config.DISCRIMINATOR_S2_FILE, config.DISCRIMINATOR_SKEWNESS_FILE, config.DISCRIMINATOR_FLATNESS_FILE]):
			self.discriminators_data[discriminator_name]["model"].load_state_dict(torch.load(models_path / file_name))