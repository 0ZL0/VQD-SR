import torch
import argparse
import yaml
import os
import sys # Ensure sys is imported
import datetime
import importlib
import torch.optim as optim

# Attempt to add taming-transformers to sys.path for easier imports
# This assumes the script is run from the project root where 'taming-transformers' is a subdirectory
try:
    # Get the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the taming-transformers directory
    taming_transformers_path = os.path.join(script_dir, "taming-transformers")
    if os.path.isdir(taming_transformers_path) and taming_transformers_path not in sys.path:
        sys.path.insert(0, taming_transformers_path)
        print(f"Added {taming_transformers_path} to sys.path for taming module resolution.")
    # The 'taming' package is expected to be directly inside 'taming-transformers' directory.
    # e.g. ./taming-transformers/taming/models/vqgan_top.py
    # So, taming_transformers_path being in sys.path allows 'import taming.models...'
except Exception as e:
    print(f"Warning: Could not automatically add taming-transformers to sys.path: {e}")
    print("Please ensure 'taming-transformers' directory is in your PYTHONPATH or installed.")

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --------------------- Utility Functions ---------------------
def load_config_from_yaml(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path, package=None)
    cls = getattr(module, class_name)
    return cls(**config.get("params", dict()))

# --------------------- Dataset Definition ---------------------
class ImageDataset(Dataset):
    def __init__(self, image_folder_path, transform=None):
        self.transform = transform
        self.image_paths = []
        allowed_extensions = {'.png', '.jpg', '.jpeg'}
        for filename in os.listdir(image_folder_path):
            if os.path.splitext(filename)[1].lower() in allowed_extensions:
                self.image_paths.append(os.path.join(image_folder_path, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a custom VQGAN model.")
    parser.add_argument('--config_stage1', type=str, required=True, help='Path to YAML config for Stage 1')
    parser.add_argument('--config_stage2', type=str, required=True, help='Path to YAML config for Stage 2')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the image dataset directory')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs_stage1', type=int, default=50, help='Number of epochs for Stage 1')
    parser.add_argument('--epochs_stage2', type=int, default=100, help='Number of epochs for Stage 2')
    parser.add_argument('--logdir', type=str, default='logs_custom_vqgan', help='Directory for TensorBoard logs and checkpoints')

    args = parser.parse_args()

    # Create log directory if it doesn't exist
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Create timestamped subdirectory for the current run
    timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_logdir = os.path.join(args.logdir, timestamp)
    os.makedirs(run_logdir)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=run_logdir)

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Define Transformations
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # Converts to [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1] range
    ])

    # Instantiate Dataset and DataLoader
    print(f"Loading dataset from: {args.dataset_path}")
    train_dataset = ImageDataset(image_folder_path=args.dataset_path, transform=transformations)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"Dataset loaded. Number of images: {len(train_dataset)}")

    # Testing DataLoader
    print("Testing DataLoader...")
    if len(train_dataloader) > 0:
        for i, batch in enumerate(train_dataloader):
            print(f"Batch {i+1}, Shape: {batch.shape}")
            if i == 2: # Print info for first 3 batches
                break
    else:
        print("DataLoader is empty. Please check dataset path and contents.")
    print("DataLoader test complete.")

    # Placeholder for Stage 1 training logic
    # train_stage1(args, train_dataloader, writer) # Will be called here

    # Placeholder for Stage 2 training logic
    # TODO: Implement Stage 2 training

    train_stage1(args, train_dataloader, writer)
    train_stage2(args, train_dataloader, writer) # Call Stage 2 after Stage 1


    writer.close()
    print(f"Training run logged to: {run_logdir}")

# --------------------- Training Stages ---------------------
def train_stage1(args, train_dataloader, writer):
    # Ensure writer.log_dir is available for constructing checkpoint path for Stage 2
    if not hasattr(writer, 'log_dir') or not writer.log_dir:
        print("Error: TensorBoard writer.log_dir is not set. Cannot proceed with Stage 1.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for Stage 1 training.")

    print("Loading Stage 1 Config...")
    stage1_config = load_config_from_yaml(args.config_stage1)

    print("Instantiating Stage 1 VQModel...")
    # Note: This assumes VQModel and its components can be instantiated and used
    # without the PyTorch Lightning Trainer. This might require the `taming-transformers`
    # library to be in the PYTHONPATH.
    try:
        model_config = stage1_config['model']
        vq_model = instantiate_from_config(model_config).to(device)
        # The loss function is typically part of the VQModel in taming-transformers
        # and is instantiated within VQModel's __init__ based on lossconfig.
        # We access it via an attribute, e.g., vq_model.loss
        if not hasattr(vq_model, 'loss') or vq_model.loss is None:
            # Fallback if loss is not directly an attribute or needs explicit setup
            # This might happen if the VQModel's __init__ doesn't create self.loss
            # without PL. For now, we assume VQModel structure from taming library.
            print("Warning: vq_model.loss is not directly available. Attempting to use lossconfig from model_config.")
            loss_fn = instantiate_from_config(model_config['params']['lossconfig']).to(device)
        else:
            loss_fn = vq_model.loss.to(device)

    except ImportError as e:
        print(f"ImportError during Stage 1 model instantiation: {e}")
        print("Please ensure the 'taming-transformers' library and its dependencies are correctly installed and accessible in PYTHONPATH.")
        print("You might need to clone the repository and add it to PYTHONPATH if it's not installed as a package.")
        print("Example: export PYTHONPATH=$PYTHONPATH:/path/to/taming-transformers")
        return # Exit if model cannot be instantiated
    except Exception as e:
        print(f"Error during Stage 1 model instantiation: {e}")
        print("This could be due to PyTorch Lightning specific initializations in the VQModel.")
        print("Consider adapting the VQModel or its components (Endecoder, Loss) for direct use.")
        return


    print("Setting up Optimizers for Stage 1...")
    lr = args.learning_rate
    # Based on VQModel's configure_optimizers in taming_models_vqgan_top.py
    # Ensure that vq_model.endecoder and loss_fn.discriminator exist and have parameters.
    if not hasattr(vq_model, 'endecoder'):
        print("Error: vq_model does not have an 'endecoder' attribute. Cannot setup optimizer.")
        return
    if not hasattr(loss_fn, 'discriminator') or not hasattr(loss_fn.discriminator, 'parameters'):
        print("Error: loss_fn.discriminator is not available or has no parameters. Cannot setup optimizer for discriminator.")
        # Depending on the loss, a discriminator might be optional.
        # For VQLPIPSWithDiscriminator, it's required.
        return

    opt_ae = optim.Adam(vq_model.endecoder.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_disc = optim.Adam(loss_fn.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    print("Starting Stage 1 Training Loop...")
    global_step = 0
    for epoch in range(args.epochs_stage1):
        print(f"Stage 1, Epoch {epoch+1}/{args.epochs_stage1}")
        vq_model.train() # Set model to training mode (affects dropout, batchnorm etc.)

        if len(train_dataloader) == 0:
            print("Warning: DataLoader is empty. Skipping training loop for this epoch.")
            continue

        for batch_idx, images in enumerate(train_dataloader):
            images = images.to(device)

            # Forward pass through VQModel
            # VQModel's forward typically returns: x_recon, vq_loss, info_dict
            # where info_dict contains {'vq_loss': vq_loss, 'commit_loss': commit_loss, 'perplexity': perplexity}
            # The exact return signature might vary; adapting based on common taming.models.vqgan.VQModel
            xrec, qloss, _ = vq_model(images) # Assuming this is the call signature for taming.models.vqgan.VQModel

            # Generator update (optimizer_idx=0)
            opt_ae.zero_grad()
            # The loss_fn is expected to be like VQLPIPSWithDiscriminator
            # Its call signature: __call__(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, split="train")
            aeloss, log_dict_ae = loss_fn(qloss, images, xrec, optimizer_idx=0, global_step=global_step,
                                          last_layer=vq_model.get_last_layer(), split="train")
            aeloss.backward()
            opt_ae.step()

            # Discriminator update (optimizer_idx=1)
            opt_disc.zero_grad()
            discloss, log_dict_disc = loss_fn(qloss, images, xrec.detach(), optimizer_idx=1, global_step=global_step,
                                              last_layer=vq_model.get_last_layer(), split="train")
            discloss.backward()
            opt_disc.step()

            if batch_idx % 10 == 0: # Log every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx}, AE Loss: {aeloss.item():.4f}, Disc Loss: {discloss.item():.4f}, QLoss: {qloss.item():.4f}")
                writer.add_scalar('Stage1/aeloss', aeloss.item(), global_step)
                writer.add_scalar('Stage1/discloss', discloss.item(), global_step)
                writer.add_scalar('Stage1/qloss', qloss.item(), global_step) # Log qloss as well

                # log_dict_ae/disc can contain tensors, ensure .item() is called
                for k, v in log_dict_ae.items():
                    if hasattr(v, 'item'):
                        writer.add_scalar(f'Stage1/ae/{k.replace("train/", "")}', v.item(), global_step)
                    else:
                        writer.add_scalar(f'Stage1/ae/{k.replace("train/", "")}', v, global_step)
                for k, v in log_dict_disc.items():
                    if hasattr(v, 'item'):
                        writer.add_scalar(f'Stage1/disc/{k.replace("train/", "")}', v.item(), global_step)
                    else:
                        writer.add_scalar(f'Stage1/disc/{k.replace("train/", "")}', v, global_step)

            global_step += 1

    print("Saving Stage 1 Model...")
    checkpoint_path = os.path.join(writer.log_dir, 'vqgan_stage1_checkpoint.pth')
    # Save the VQModel's state_dict. This should include the endecoder, quantizer, and loss (if loss has state like discriminator).
    torch.save(vq_model.state_dict(), checkpoint_path)
    print(f"Stage 1 Model saved to {checkpoint_path}")

    print("Stage 1 Training Complete.")


def train_stage2(args, train_dataloader, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for Stage 2 training.")

    # Ensure writer.log_dir is available for constructing checkpoint path
    if not hasattr(writer, 'log_dir') or not writer.log_dir:
        print("Error: TensorBoard writer.log_dir is not set. Cannot proceed with Stage 2.")
        return

    stage1_checkpoint_path = os.path.join(writer.log_dir, 'vqgan_stage1_checkpoint.pth')
    if not os.path.exists(stage1_checkpoint_path):
        print(f"Error: Stage 1 checkpoint not found at {stage1_checkpoint_path}. Run Stage 1 first.")
        return

    print("Loading Stage 2 Config...")
    stage2_config = load_config_from_yaml(args.config_stage2)
    stage2_model_config = stage2_config['model']

    # Ensure 'params' key exists in stage2_model_config, initialize if not
    if 'params' not in stage2_model_config:
        stage2_model_config['params'] = {}

    print(f"Setting Stage 1 checkpoint for Stage 2 model: {stage1_checkpoint_path}")
    stage2_model_config['params']['top_ckpt_path'] = stage1_checkpoint_path
    # Ensure 'ckpt_path' (usually for resuming training of the same model) is None or not set
    # to avoid conflict with top_ckpt_path which is for loading a different model's weights.
    stage2_model_config['params']['ckpt_path'] = None


    print("Instantiating Stage 2 Multi-Scale VQModel...")
    try:
        # The target class is likely taming.models.vqgan_multi_scale_load_top_scale.VQModel
        vq_model_ms = instantiate_from_config(stage2_model_config).to(device)
        if not hasattr(vq_model_ms, 'loss') or vq_model_ms.loss is None:
            print("Warning: vq_model_ms.loss is not directly available. Attempting to use lossconfig from model_config.")
            # This assumes lossconfig is under model_config['params'] if not part of VQModel_MS directly
            loss_fn_ms = instantiate_from_config(stage2_model_config['params']['lossconfig']).to(device)
        else:
            loss_fn_ms = vq_model_ms.loss.to(device)
    except ImportError as e:
        print(f"ImportError during Stage 2 model instantiation: {e}")
        print("Please ensure the 'taming-transformers' library is in your PYTHONPATH.")
        return
    except Exception as e:
        print(f"Error instantiating Stage 2 model or loss: {e}")
        # This can happen if top_ckpt_path is not handled correctly by the model's __init__
        # or if other parameters are missing/incorrect.
        return

    print("Setting up Optimizers for Stage 2...")
    lr = args.learning_rate
    # top_lr_weight is a parameter in the config for taming.models.vqgan_multi_scale_load_top_scale.VQModel
    # It's used to scale the learning rate for the top-level parameters of the VQGAN.
    # Defaulting to 1.0 if not specified in the config for robustness.
    top_lr_weight = stage2_model_config['params'].get('top_lr_weight', 1.0)
    lr_top = lr * top_lr_weight
    print(f"Base LR: {lr}, Top LR (for loaded top scale): {lr_top}")

    if not hasattr(vq_model_ms, 'endecoder') or not hasattr(vq_model_ms.endecoder, 'top'):
        print("Error: Multi-scale model structure (endecoder.top) not found. Cannot set up optimizers.")
        return
    if not hasattr(loss_fn_ms, 'discriminator') or not hasattr(loss_fn_ms.discriminator, 'parameters'):
        print("Error: loss_fn_ms.discriminator not found or has no parameters.")
        # Consider if the loss for stage 2 might optionally not have a discriminator
        return

    # Parameter separation for different learning rates (copied from VQModel.configure_optimizers)
    # This assumes vq_model_ms.endecoder.top are the parameters loaded from stage 1
    top_scale_params_ids = list(map(id, vq_model_ms.endecoder.top.parameters()))
    base_params = [p for p in vq_model_ms.endecoder.parameters() if id(p) not in top_scale_params_ids]

    # Ensure all parameters in endecoder are covered
    all_endecoder_params_ids = list(map(id, vq_model_ms.endecoder.parameters()))
    if not set(all_endecoder_params_ids).issubset(set(map(id, base_params)) | set(top_scale_params_ids)) :
        print("Warning: Some parameters in endecoder might not be included in the optimizer.")

    ae_params_ms = []
    if base_params: # Add base parameters only if they exist
        ae_params_ms.append({'params': base_params, 'lr': lr})

    # Add top parameters, ensuring they exist
    if list(vq_model_ms.endecoder.top.parameters()):
         ae_params_ms.append({'params': vq_model_ms.endecoder.top.parameters(), 'lr': lr_top})
    else:
        print("Warning: vq_model_ms.endecoder.top has no parameters. Top LR will not be applied to any specific group.")


    if not ae_params_ms:
        print("Error: No parameters found for the autoencoder optimizer in Stage 2.")
        return

    opt_ae_ms = optim.Adam(ae_params_ms, lr=lr, betas=(0.5, 0.9)) # Base lr is default for Adam
    opt_disc_ms = optim.Adam(loss_fn_ms.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    print("Starting Stage 2 Training Loop...")
    global_step_stage2 = 0 # Separate global step for stage 2 logging
    for epoch in range(args.epochs_stage2):
        print(f"Stage 2, Epoch {epoch+1}/{args.epochs_stage2}")
        vq_model_ms.train()

        if len(train_dataloader) == 0:
            print("Warning: DataLoader is empty. Skipping training loop for this epoch.")
            continue

        for batch_idx, images in enumerate(train_dataloader):
            images = images.to(device)

            # Forward pass for multi-scale VQGAN
            # Expected to return x_recon, vq_loss, info_dict (similar to single-scale)
            xrec_ms, qloss_ms, _ = vq_model_ms(images)

            # Generator update
            opt_ae_ms.zero_grad()
            aeloss_ms, log_dict_ae_ms = loss_fn_ms(qloss_ms, images, xrec_ms, optimizer_idx=0,
                                                  global_step=global_step_stage2,
                                                  last_layer=vq_model_ms.get_last_layer(), split="train")
            aeloss_ms.backward()
            opt_ae_ms.step()

            # Discriminator update
            opt_disc_ms.zero_grad()
            discloss_ms, log_dict_disc_ms = loss_fn_ms(qloss_ms, images, xrec_ms.detach(), optimizer_idx=1,
                                                      global_step=global_step_stage2,
                                                      last_layer=vq_model_ms.get_last_layer(), split="train")
            discloss_ms.backward()
            opt_disc_ms.step()

            if batch_idx % 10 == 0: # Log every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx}, AE Loss: {aeloss_ms.item():.4f}, Disc Loss: {discloss_ms.item():.4f}, QLoss: {qloss_ms.item():.4f}")
                writer.add_scalar('Stage2/aeloss', aeloss_ms.item(), global_step_stage2)
                writer.add_scalar('Stage2/discloss', discloss_ms.item(), global_step_stage2)
                writer.add_scalar('Stage2/qloss', qloss_ms.item(), global_step_stage2)
                # log_dict_ae/disc can contain tensors, ensure .item() is called
                for k, v in log_dict_ae_ms.items():
                    if hasattr(v, 'item'):
                         writer.add_scalar(f'Stage2/ae/{k.replace("train/", "")}', v.item(), global_step_stage2)
                    else:
                         writer.add_scalar(f'Stage2/ae/{k.replace("train/", "")}', v, global_step_stage2)
                for k, v in log_dict_disc_ms.items():
                    if hasattr(v, 'item'):
                        writer.add_scalar(f'Stage2/disc/{k.replace("train/", "")}', v.item(), global_step_stage2)
                    else:
                        writer.add_scalar(f'Stage2/disc/{k.replace("train/", "")}', v, global_step_stage2)
            global_step_stage2 += 1

    print("Saving Stage 2 Model...")
    final_checkpoint_path = os.path.join(writer.log_dir, 'vqgan_stage2_final.pth')
    # Save the VQModel_MS state_dict.
    torch.save(vq_model_ms.state_dict(), final_checkpoint_path)
    print(f"Stage 2 Model saved to {final_checkpoint_path}")
    print("Stage 2 Training Complete.")


if __name__ == '__main__':
    main()
