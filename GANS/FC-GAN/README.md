# FC-GAN: MNIST Digit Generator

A simple yet effective Fully Connected Generative Adversarial Network that learns to create handwritten digits from scratch.

## What is a GAN?

Imagine two AI networks in an endless game:
- **Generator** (The Forger): Creates fake handwritten digits
- **Discriminator** (The Detective): Spots fake vs real digits

As they compete, both get better. Eventually, the Generator becomes so skilled that even the Detective can't tell its creations from real handwriting.

## Network Architecture

**Generator Network**
```
Random Noise (128) → FC Layers → Batch Norm → LeakyReLU → Output (784)
```
Transforms random noise into 28×28 digit images

**Discriminator Network** 
```
Image (784) → FC Layers → LeakyReLU → Dropout → Real/Fake (1)
```
Judges whether an image is authentic or generated

## Training Process

- **Dataset**: 60,000 MNIST handwritten digits
- **Training Time**: 400 epochs of adversarial competition
- **Key Features**: 
  - Smart learning rate scheduling
  - Label smoothing for stability
  - Gradient clipping to prevent chaos

## Evolution Over Time (Expected)

| Stage | What Happens |
|-------|-------------|
| **Epoch 1** | Generator produces complete noise - unrecognizable patterns |
| **Epochs 10-50** | Faint blob-like structures begin to appear |
| **Epochs 50-100** | Digit-like shapes emerge but lack clear definition |
| **Epochs 100-150** | Recognizable digits form with improved clarity |
| **Epochs 150-200** | High-quality digits with proper proportions achieved |
| **Epoch 200+** | Peak performance - realistic handwritten digits indistinguishable from real ones |


## Project Structure

```
FC-GAN/
├── Simple_GAN.ipynb          # Main training notebook
├── best_gen_epoch_001.pth    # Trained generator
├── generated_images/         # Training progression
└── README.md
```

## Why This Matters

This project demonstrates how two simple networks, through competition alone, can learn to create realistic data without ever being explicitly taught what makes a "good" digit. It's artificial creativity in its purest form.

---

*Built with PyTorch • Trained on MNIST • 400 epochs of adversarial learning*