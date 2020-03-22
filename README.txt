The following are the most relevant files for our project. Other files in the folder can be ignored for the purposes of this project.

* classifier.py, decoder.py, encoder.py, merger.py, and refiner.py in models folder the are the backbone pytorch modules for the pix2vox architecture.

* test.py and train.py in the core folder are responsible for testing and training models, respectively.

* CORAL_epoch_manager.py, DANN_epoch_manager.py, and voxel_cls_epoch_manager.py implement domain adaptation. CORAL and DANN borrow open source code from github repository, and is cited as comments in those files. voxel_cls_epoch_manager.py is our proposed architecture.

* The datasets folder contains .json files which detail the test/train/val split of the relevant datasets.

* The caches folder is contains caches just to speed up data loading.

* The utils folder contains various utilities used in training/testing. binvox_converter.py, binvox_rw.py, and binvox_visualization.py help with working with voxels. data_loaders.py helps load in datasets. data_transforms.py contains data augmentation functions. gradient_reversal_module.py implements gradient reversal.

* config.py contains configuration settings for training/testing modules

* runner.py is the executable python script that enables training/testing.