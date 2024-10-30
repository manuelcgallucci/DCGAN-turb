import torch
from torch.utils.data import DataLoader

import config
import dataloader as dl
from utility import get_structure_scales, get_uuid, save_meta

# from train_environment import TrainingEnvironment as TrainEnvironment
from train_environment_generator_with_structures import TrainingEnvironmentGeneratorWithStrucutres as TrainEnvironment

def main():
    continue_model_uuid = "Bf3XOK"
    model_uuid = get_uuid()
    out_dir = config.OUTPUT_PATH / model_uuid

    data_type = "full" # full samples from the original data 
    batch_size = 32
    data_stride = 2**15
    len_samples = 2**15

    train_dataset, _, _ = dl.loadDataset(type=data_type, stride=data_stride, len_samples=len_samples)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)

    lr = 0.002
    epochs = 120
    beta = 1.0 # generator loss multiplier 0.5
    k_epochs_d = 2
    k_epochs_g = 1
    structure_scales=get_structure_scales()
    weights = torch.Tensor([1,1,0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25])
    checkpoint_epochs = 20

    training_environment = TrainEnvironment(epochs, lr)
    if continue_model_uuid is not None:
        training_environment.load_from_checkpoint(config.OUTPUT_PATH / continue_model_uuid)

    if checkpoint_epochs is not None:
        out_dir.mkdir(exist_ok=False)
    train_time = training_environment.train(dataloader_train, structure_scales, weights, k_epochs_g, k_epochs_d, beta, checkpoint_epochs, out_dir, verbose=True)

    if checkpoint_epochs is None:
        out_dir.mkdir(exist_ok=False)
    training_environment.save_models(out_dir)
    training_environment.save_metadata(out_dir)
    
    save_meta(out_dir / config.METADATA_FILE, {
        "model_uuid": model_uuid,
        "lr": lr,
        "epochs": epochs,
        "checkpoint_epochs": checkpoint_epochs,
        "beta": beta,
        "k_epochs_d": k_epochs_d,
        "k_epochs_g": k_epochs_g,
        "batch_size": batch_size,
        "data_stride": data_stride,
        "len_samples": len_samples,
        "train_time": train_time,
        "train_environment": str(training_environment.__class__.__name__),
        "continue_model_uuid": continue_model_uuid,
        "structure_scales": structure_scales.tolist(),
        "weights": weights.tolist()
    })

if __name__ == '__main__':
    main()