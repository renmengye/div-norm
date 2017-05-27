# div-norm
Implementation of divisive normalization in TensorFlow

## CIFAR experiments
- First download the python version of CIFAR dataset here. https://www.cs.toronto.edu/~kriz/cifar.html

  - For CIFAR-10:
  Create a folder named "cifar-10" and move the uncompressed folder "cifar-10-batches-py" into "cifar-10".

  - For CIFAR-100:
  Create a folder named "cifar-100" and move the uncompressed folder "cifar-100-py" into "cifar-100".

- Run training and testing
  ```
  python run_cifar_exp.py --dataset {cifar-10/cifar-100} --model {MODEL} --verbose
  ```

  Replace MODEL with a pre-configured model name, e.g. "dn". For details, please take a look at cifar_exp_config.py.


## Super-resolution experiments

* Download the datasets [Set5, Set14, BSD200](https://github.com/huangzehao/Super-Resolution.Benckmark). Create a folder named "sr_data" and put uncompressed datasets as subfolders in it. Note that our code depends on ```h5py```, ```cv2```.

* Generate the training data by running ```gen_sr_data.m``` in Matlab (we used matlab's imresize function to generate training data which is named as, e.g., "data_X4.h5" in folder "sr_data"). You can easily modify the script to use your own training or testing data.

* Run the following command to train and test the model. Please refer to the documentation in the beginning of file for more on the configurations.

  ```
  python run_sr_exp.py --model dnms --data_folder sr_data --results results --verbose
  ```