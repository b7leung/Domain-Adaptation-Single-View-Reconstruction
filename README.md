# Overview

The original code of this repository comes from the code for pix2vox: 
https://github.com/hzxie/Pix2Vox

It has been modified in the following ways:
* Easier to view reconstructed voxels; after evaluation, they are all outputted into a trial-specific folder
* Compatibility with ODDS datasets
* Compatibility for domain adaptation experiments (eg, load/train on the source and target domain datasets)
* Refactor the main testing loop into polymorphic "epoch_managers", to enable more modular and flexible experimentation for different architecures. The main advantage is we keep the same code for the main reconstruction encoder/decoders, dataloading, and validation set testing.
* tqdm progress bars
* Generally more readable, more efficient code

# Important Notes

In order create your own variant of pix2vox, a new __EpochManager__ must be created. Using polymorphism, this gives you the benefit of being able to try aribtarily many variants of pix2vox with different losses, domain adaptation techniques, etc without having to copy/paste the boilerplate code, which is a bad coding practice. In general, an EpochManager is responsible for the minibatch update steps in each epoch and needs to follow the following interface:

* `init_epoch()` method which initalizes things before epochs (eg calling .train() on any variant-specific neural network)
* a `perform_step(self, source_batch_data, target_batch_data, epoch_idx)` method which actually performs the minibatch update step. It also needs to return a dict meant to be a row in a pandas dataframe which records information for this current minibatch step
* an `end_epoch()` method which is called after epochs (eg calling .step() for any of your schedulers)

Please see the existing EpochManagers (core/CORAL_epoch_manager.py, core/DANN_epoch_manager.py, core/noDA_epoch_manager.py, core/voxel_cls_epoch_manager.py) as an example.


# Example Usage

Note that the code needs python 3.7, matplotlib 3.0.3, and a bunch of other dependencies. If you are running on the SVCL gpu cluster, just use __conda activate standard__. Otherwise, the dependencies are argparse, easydict, matplotlib, numpy, opencv-python, scipy, pytorch, torchvision, pandas, tqdm, sklearn.


## Training

The following command trains the Pix2Vox with DANN domain adaptation, and a voxel classifier from scratch for 2 epochs. Source dataset is ShapeNet, target dataset is OOWL. The classes are aeroplane, car, display, lamp, telephone, watercraft (these are the only classes that overlap with our version of ShapeNet and ODDS). Change the GPU number as necessary.

`python runner.py --gpu 3 --batch_size 64 --trial_comment ExampleTrial --train_target_dataset OOWL --DA VoxelCls --voxel_lam -2 --DANN_lam 1 --epoch 2 --classes aeroplane car display lamp telephone watercraft`

After training is done, you should a folder for that trial in output/timestamp_ExampleTrial_TRAIN. Inside, there will be a file called run_cfg_info.txt which tells you the parameters that you used to run it. Also, the weights are saved into a checkpoints folder. Finally, training_record.pkl contains logged data during the training, in the form of a pandas dataframe. Usage and visualizations for the dataframe can be found in notebooks/View_Training_Record.ipynb.

## Testing

To test the model using pretrained weights from above:

`python runner.py --gpu 1 --weights ./output/2020_03_26--21_40_39_ExampleTrial_TRAIN/checkpoints/best-ckpt.pth --trial_comment ExampleTrial --test --test_dataset OOWL --save_num -1 --classes aeroplane car display lamp telephone watercraft`

The save_num -1 option indicates that you want to save all the reconstructions. They will be in output/timestamp_ExampleTrial_TEST.
