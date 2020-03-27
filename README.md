# Overview

The original code of this repository comes from the code for pix2vox: 
https://github.com/hzxie/Pix2Vox

It has been modified in the following ways:
* Easier to view reconstructed voxels; after evaluation, they are all outputted into a trial-specific folder
* Compatibility for domain adaptation experiments (eg, load/train on the source and target domain datasets)
* Refactor the main testing loop into polymorphic "epoch_managers", to enable more modular and flexible experimentation for different architecures. The main advantage is we keep the same code for the main reconstruction encoder/decoders, dataloading, and validation set testing.
* Generally more readable, more efficient code

# Important Python Modules

# Command Line arguments

# Example Usage

Note that the code needs python 3.7, matplotlib 3.0.3. 

## Training

## Testing
