#this file contains all the default parameters for the MNIST MADGAN implementation

ARGS = {
    'gpu': 1,#the number of gpu(s) to be used for training...0 in the case of only cpu
    'num_channels':3,#the number of channels in the images of the real image dataset
    'image_size':64,#the dimension to which the input images are resized for the purpose of training
    'leaky_slope':0.2,#the negative slope of the leaky ReLU activation used in the architecture
    'dataroot':'./data',#the path of the directory whose sub directories contain the image data
    'n_z':100, #the size of the noise vector which is input to the generator
    'batch_size':120, #batch size to be used while training
    'num_workers':2,#number of workers used while loading the data
    'num_generators':3,#the number of generators that are being used
    'degan':True, #use the modified DEGAN loss function
    'conv_weights_init_mean':0.0,#the mean of the normal weight initialization of the conv layers
    'conv_weights_init_dev':0.02,#the standard deviation of the normal weight initialization of the conv layers
    'bn_weights_init_mean':1.0, #the mean of the normal weight initialization of the batch norm layers
    'bn_weights_init_dev' :0.02, #the standard deviation of the
    'bn_bias_weights_init':0.0, #the constant value init of the bias weights of the bn layers
    'sharing':0,
    'gpu_add':0,
}