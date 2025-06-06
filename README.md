# QuDiffuse

QuDiffuse implements the quantum-assisted diffusion model described in `paper.tex`.
The autoencoder maps images to binary latent codes. A Deep Belief Network (DBN) is
trained using Contrastive Divergence to model these codes. Denoising is cast as a
QUBO problem solved with `dwave-neal` during sampling.

## Requirements
```
pip install -r requirements.txt
```

## Training
```
python qudiffuse.py --epochs 10 --dataset mnist
```
This trains the binary autoencoder and DBN using purely classical methods.

## Generation
After training, generate images with quantum-assisted sampling:
```
python qudiffuse.py --dataset mnist --generate --timesteps 100 --window 5
```
This uses the Neal simulator to solve the joint QUBO over multiple timesteps.
Generated images are saved to `generated.png`.

