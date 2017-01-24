# div-norm
Implementation of divisive normalization in TensorFlow

## CIFAR experiments
First download the python version of CIFAR dataset here.
https://www.cs.toronto.edu/~kriz/cifar.html

For CIFAR-10:
Create a folder named "cifar-10" and move the uncompressed folder 
"cifar-10-batches-py" into "cifar-10".

For CIFAR-100:
Create a folder named "cifar-100" and move the uncompressed folder 
"cifar-100-py" into "cifar-100".

```
python run_cifar_exp.py --dataset {cifar-10/cifar-100} --model {MODEL} --verbose
```

Replace MODEL with a pre-configured model name, e.g. "dn". For details, please
take a look at cifar_exp_config.py.
