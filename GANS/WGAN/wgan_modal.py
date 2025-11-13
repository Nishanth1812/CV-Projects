"""
WGAN with Gradient Penalty - Modal Implementation
Trains a Wasserstein GAN with gradient penalty on CelebA dataset.
"""

import modal
from pathlib import Path

# Define Modal app
app = modal.App("wgan-training")

# Create Modal image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "pillow",
    "tqdm",
    "kagglehub",
    "numpy",
)

# Create Modal volumes for persistent storage
dataset_volume = modal.Volume.from_name("celeba-dataset", create_if_missing=True)
models_volume = modal.Volume.from_name("wgan-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("wgan-outputs", create_if_missing=True)

# Local model paths
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
LOCAL_GEN_MODEL = SCRIPT_DIR / "best_generator.pth"
LOCAL_CRITIC_MODEL = SCRIPT_DIR / "best_critic.pth"

# Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
MAX_GRAD_NORM = 1.0


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/data": dataset_volume,
        "/models": models_volume,
        "/outputs": outputs_volume,
    },
    timeout=86400,
    secrets=[modal.Secret.from_name("kaggle-secret")],
)
def train_wgan():
    """Main training function for WGAN with gradient penalty"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    from tqdm import tqdm
    import os
    import time as time_module
    import torchvision.utils as vutils
    import kagglehub #type: ignore
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Helper function to create comparison images
    def create_comparison_image(real_images, fake_images, epoch, save_path):
        """Create side-by-side comparison of real and fake images"""
        real_grid = torchvision.utils.make_grid(real_images, normalize=True, nrow=8, padding=2)
        fake_grid = torchvision.utils.make_grid(fake_images, normalize=True, nrow=8, padding=2)
        
        real_np = (real_grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        fake_np = (fake_grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        real_img = Image.fromarray(real_np)
        fake_img = Image.fromarray(fake_np)
        
        label_height = 40
        comparison = Image.new('RGB', (real_img.width + fake_img.width, max(real_img.height, fake_img.height) + label_height), 'white')
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((real_img.width // 2 - 50, 10), f"REAL (Epoch {epoch})", fill='black', font=font)
        draw.text((real_img.width + fake_img.width // 2 - 50, 10), f"FAKE (Epoch {epoch})", fill='black', font=font)
        
        comparison.paste(real_img, (0, label_height))
        comparison.paste(fake_img, (real_img.width, label_height))
        comparison.save(save_path)
    
    # Download CelebA dataset
    print("Downloading CelebA dataset from Kaggle...")
    dataset_path = "/data/celeba"
    img_folder_path = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")
    
    if not os.path.exists(img_folder_path):
        path = kagglehub.dataset_download("jessicali9530/celeba-dataset")  # type: ignore
        print(f"Dataset path: {path}")
        
        import shutil
        os.makedirs(dataset_path, exist_ok=True)
        
        source_img_path = os.path.join(path, "img_align_celeba")
        dest_img_path = os.path.join(dataset_path, "img_align_celeba")
        
        if os.path.exists(source_img_path):
            shutil.copytree(source_img_path, dest_img_path)
        
        dataset_volume.commit()
        print("Dataset downloaded and saved")
    else:
        print("Dataset already exists")
    
    # Gradient Penalty Function
    def gradient_penalty(critic, real, fake, device="cpu"):
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * alpha + fake.detach() * (1 - alpha)
        interpolated_images.requires_grad_(True)
        
        mixed_scores = critic(interpolated_images)
        
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        return torch.mean((gradient_norm - 1) ** 2)
    
    # Discriminator (Critic) Model
    class Discriminator(nn.Module):
        def __init__(self, channels_img, features_d):
            super().__init__()
            self.disc = nn.Sequential(
                nn.Conv2d(channels_img, features_d, 4, 2, 1),
                nn.LeakyReLU(0.2),
                self._block(features_d, features_d * 2, 4, 2, 1),
                self._block(features_d * 2, features_d * 4, 4, 2, 1),
                self._block(features_d * 4, features_d * 8, 4, 2, 1),
                nn.Conv2d(features_d * 8, 1, 4, 2, 0),
            )

        def _block(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2),
            )

        def forward(self, x):
            return self.disc(x)
    
    # Generator Model
    class Generator(nn.Module):
        def __init__(self, channels_noise, channels_img, features_g):
            super().__init__()
            self.net = nn.Sequential(
                self._block(channels_noise, features_g * 16, 4, 1, 0),
                self._block(features_g * 16, features_g * 8, 4, 2, 1),
                self._block(features_g * 8, features_g * 4, 4, 2, 1),
                self._block(features_g * 4, features_g * 2, 4, 2, 1),
                nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),
                nn.Tanh(),
            )

        def _block(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)
    
    # Weight Initialization
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    # Custom dataset class to handle corrupted images
    class RobustImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f"⚠ Skipping corrupted image at index {index}")
                return self.__getitem__((index + 1) % len(self))
    
    # Dataset Transforms
    transforms_pipeline = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * CHANNELS_IMG, [0.5] * CHANNELS_IMG),
    ])
    
    # Setup DataLoader
    print("Setting up DataLoader...")
    loader = DataLoader(
        RobustImageFolder(root=dataset_path, transform=transforms_pipeline),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    
    # Initialize models
    print("Initializing models...")
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)
    
    # Setup optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)
    gen.train()
    critic.train()
    
    # Setup directories
    checkpoints_dir = "/models/checkpoints"
    outputs_dir = "/outputs"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    best_gen_loss = float('inf')
    best_critic_loss = float('inf')
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
    best_gen_path = "/models/best_generator.pth"
    best_critic_path = "/models/best_critic.pth"
    
    def get_model_state_dict(model):
        return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    def load_model_state_dict(model, state_dict):
        # Remove _orig_mod. prefix if present (from torch.compile)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    # Resume from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\n{'='*60}")
        print("📦 RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_state_dict(gen, checkpoint['gen_state'])
        load_model_state_dict(critic, checkpoint['critic_state'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state'])
        opt_critic.load_state_dict(checkpoint['opt_critic_state'])
        start_epoch = checkpoint['epoch']
        best_gen_loss = checkpoint['best_gen_loss']
        best_critic_loss = checkpoint['best_critic_loss']
        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"{'='*60}\n")
    elif os.path.exists(best_gen_path) and os.path.exists(best_critic_path):
        print(f"\n{'='*60}")
        print("📦 LOADING BEST MODELS")
        print(f"{'='*60}")
        try:
            load_model_state_dict(gen, torch.load(best_gen_path, map_location=device))
            load_model_state_dict(critic, torch.load(best_critic_path, map_location=device))
            print("✓ Loaded best models")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("⚠ Model architecture mismatch - starting fresh")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("🆕 STARTING FRESH TRAINING")
        print(f"{'='*60}\n")
    
    # Training configuration
    epoch_times = []
    training_start_time = time_module.time()
    
    print(f"\n{'='*60}")
    print(f"Device: {device} | Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE} | Epochs: {NUM_EPOCHS}")
    print(f"Starting from epoch: {start_epoch}")
    print(f"{'='*60}\n")
    
    # Training Loop with FP32 for Maximum Stability
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time_module.time()
        epoch_gen_losses = []
        epoch_critic_losses = []
        batch_times = []  # Reset batch times for this epoch
        
        for batch_idx, (real, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            try:
                batch_start = time_module.time()
                real = real.to(device, non_blocking=True)
                cur_batch_size = real.shape[0]
                
                # Skip incomplete batches
                if cur_batch_size != BATCH_SIZE:
                    continue

                # Train Critic with Mixed Precision - Updated API
                critic_loss_sum = 0
                valid_critic_iters = 0
                
                for _ in range(CRITIC_ITERATIONS):
                    noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device, non_blocking=True)
                    fake = gen(noise)
                    critic_real = critic(real).reshape(-1)
                    critic_fake = critic(fake.detach()).reshape(-1)
                    
                    gp = gradient_penalty(critic, real, fake, device=device)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                    
                    if torch.isnan(loss_critic) or torch.isinf(loss_critic):
                        continue
                    
                    critic.zero_grad()
                    loss_critic.backward()
                    
                    has_nan = any(
                        param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                        for param in critic.parameters()
                    )
                    
                    if has_nan:
                        opt_critic.zero_grad()
                        continue
                    
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                    opt_critic.step()
                    
                    critic_loss_sum += loss_critic.item()
                    valid_critic_iters += 1

                if valid_critic_iters == 0:
                    continue
                
                avg_critic_loss = critic_loss_sum / valid_critic_iters
                epoch_critic_losses.append(avg_critic_loss)
                batch_times.append(time_module.time() - batch_start)

                # Train Generator
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device, non_blocking=True)
                fake = gen(noise)
                gen_fake = critic(fake).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                
                if torch.isnan(loss_gen) or torch.isinf(loss_gen):
                    continue
                
                gen.zero_grad()
                loss_gen.backward()
                
                has_nan = any(
                    param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    for param in gen.parameters()
                )
                
                if has_nan:
                    opt_gen.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(gen.parameters(), MAX_GRAD_NORM)
                opt_gen.step()

                epoch_gen_losses.append(loss_gen.item())
            except Exception as e:
                print(f"\n⚠ Error in batch {batch_idx}: {e}")
                continue
                
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} | "
                      f"D: {avg_critic_loss:.4f}, G: {loss_gen:.4f} | {avg_batch_time:.3f}s/batch")

        # Calculate average losses
        if len(epoch_gen_losses) == 0 or len(epoch_critic_losses) == 0:
            print(f"\n⚠ Warning: Epoch {epoch} had no successful iterations - continuing...")
            continue
        
        avg_gen_loss = sum(epoch_gen_losses) / len(epoch_gen_losses)
        avg_critic_loss = sum(epoch_critic_losses) / len(epoch_critic_losses)
        avg_batch_time = sum(batch_times) / len(batch_times) if len(batch_times) > 0 else 0
        epoch_time = time_module.time() - epoch_start_time

        # Generate and save comparison images
        with torch.no_grad():
            fake = gen(fixed_noise)
            real_batch = next(iter(loader))[0][:64].to(device)
            
            create_comparison_image(real_batch, fake, epoch, 
                                  os.path.join(outputs_dir, f"comparison_epoch_{epoch:03d}.png"))
            
            fake_grid = torchvision.utils.make_grid(fake, normalize=True, nrow=8)
            real_grid = torchvision.utils.make_grid(real_batch, normalize=True, nrow=8)
            vutils.save_image(fake_grid, os.path.join(outputs_dir, f"fake_epoch_{epoch:03d}.png"))
            vutils.save_image(real_grid, os.path.join(outputs_dir, f"real_epoch_{epoch:03d}.png"))
        
        # Save models
        gen_state = get_model_state_dict(gen)
        critic_state = get_model_state_dict(critic)
        
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            torch.save(gen_state, best_gen_path)
            models_volume.commit()
        
        if avg_critic_loss < best_critic_loss:
            best_critic_loss = avg_critic_loss
            torch.save(critic_state, best_critic_path)
            models_volume.commit()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'gen_state': gen_state,
            'critic_state': critic_state,
            'opt_gen_state': opt_gen.state_dict(),
            'opt_critic_state': opt_critic.state_dict(),
            'best_gen_loss': best_gen_loss,
            'best_critic_loss': best_critic_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        models_volume.commit()
        outputs_volume.commit()
        
        # Print progress
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta_hours = (NUM_EPOCHS - (epoch + 1)) * avg_epoch_time / 3600
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | G: {avg_gen_loss:.4f} | D: {avg_critic_loss:.4f}")
        print(f"⏱️  Time: {epoch_time/60:.1f}m | Avg batch: {avg_batch_time:.3f}s | ETA: {eta_hours:.1f}h\n")
    
    # Final save
    models_volume.commit()
    outputs_volume.commit()
    
    total_time = (time_module.time() - training_start_time) / 3600
    print(f"\n{'='*60}")
    print(f"✓ Training complete! Total: {total_time:.1f}h")
    print(f"{'='*60}\n")
    
    return {
        "best_gen_loss": best_gen_loss,
        "best_critic_loss": best_critic_loss,
        "total_epochs": NUM_EPOCHS,
        "total_time_hours": total_time,
    }


@app.function(image=image, volumes={"/models": models_volume}, timeout=300)
def upload_local_models():
    """Upload local models to Modal volume"""
    import os
    import shutil
    
    uploaded = []
    local_gen = "/root/best_generator.pth"
    local_critic = "/root/best_critic.pth"
    
    if os.path.exists(local_gen):
        shutil.copy(local_gen, "/models/best_generator.pth")
        uploaded.append("best_generator.pth")
        print("✓ Uploaded best_generator.pth")
    
    if os.path.exists(local_critic):
        shutil.copy(local_critic, "/models/best_critic.pth")
        uploaded.append("best_critic.pth")
        print("✓ Uploaded best_critic.pth")
    
    return uploaded


@app.local_entrypoint()
def main():
    """Entry point for Modal app"""
    import signal
    import sys
    
    function_call = None
    
    def signal_handler(sig, frame):
        print("\n🛑 Stopping...")
        if function_call:
            try:
                function_call.cancel()
            except:
                pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*60)
    print("📤 UPLOADING LOCAL MODELS")
    print("="*60)
    
    if LOCAL_GEN_MODEL.exists() or LOCAL_CRITIC_MODEL.exists():
        try:
            with models_volume.batch_upload() as batch:
                if LOCAL_GEN_MODEL.exists():
                    try:
                        batch.put_file(str(LOCAL_GEN_MODEL), "/best_generator.pth")
                        print(f"✓ Uploaded {LOCAL_GEN_MODEL.name}")
                    except Exception as e:
                        if "already exists" not in str(e):
                            raise
                
                if LOCAL_CRITIC_MODEL.exists():
                    try:
                        batch.put_file(str(LOCAL_CRITIC_MODEL), "/best_critic.pth")
                        print(f"✓ Uploaded {LOCAL_CRITIC_MODEL.name}")
                    except Exception as e:
                        if "already exists" not in str(e):
                            raise
        except Exception as e:
            if "already exists" not in str(e):
                print(f"⚠ Upload error: {e}")
    else:
        print("ℹ No local models found - starting fresh")
    
    print("\nStarting WGAN training...\n")
    
    try:
        function_call = train_wgan.spawn()
        result = function_call.get()
        print(f"\n✓ Training Results:")
        print(f"  Best Gen Loss: {result['best_gen_loss']:.4f}")
        print(f"  Best Critic Loss: {result['best_critic_loss']:.4f}")
        print(f"  Total Time: {result['total_time_hours']:.2f}h")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted")
        if function_call:
            try:
                function_call.cancel()
            except:
                pass
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
