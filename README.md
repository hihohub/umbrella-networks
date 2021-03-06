# umbrella-networks
convolutional neural network system for hierarchically ordered images
- umbrella networks are not new...other researchers have investigated similar techniques...most recently, Hinton built umbrella-type networks for Google
- - https://arxiv.org/abs/1503.02531
- - https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf
- description
- - umbrella networks are an ensembling technique...the models used have been built with tensorflow
- - the umbrella network framework has different uses
- - - simplify reading image files from a hierarchical file structure
- - - construct hierarchical ensembles that predict with traversal by maximum
- - - construct joint probability ensembles
- - - test different ensemble architectures to find what's right for your data
- - - build a regular flat softmax from the bottom folders of a hierarchical file structure
- instructions
- - place all of your images into a root folder
- - organize your images into subfolders by class
- - if desired, organize those subfolders into higher level folders
- - use as many levels as you like
- - all folders must have unique names
- - when initializing the umbrella network, set levels = 0 for a regular softmax over all classes (the program ignores the higher folders), set levels = 1 to predict just the outer level clusters, set levels > 1 to predict first the outer level cluster, then the specific inner layer class, i.e. the outer level folders each contain subfolders, one for each class
- - the subfolders may also contain their own subfolders, and the umbrella network predicts as many levels as you specify in the 'levels' setting
- - if levels is non-zero and set to less than the number of layers of folders, the program compresses the lower level folders into the upper level folders, keeping all of the images
- - for example, you may set levels = 0 or 1 for any folder structure
- classes
- - Umbrella Network is the main class
- - Joint Probability Network is an ensemble of two umbrella networks, upper level and lower level, whose predicted probabilities are multiplied
- network architecture
- - although an umbrella network contains multiple models, each of them has the exact same architecture
- - two types of umbrella network architectures are available, 'sigmoid' and 'softmax'
- - for levels = 0 or 1, 'softmax' architecture makes a single model that predicts multiple classes
- - if levels > 1, 'softmax' makes a single model for the outer level, and a single model for each sublevel specified by the outer level
- - if those levels also have inner levels, and levels > 2, then further softmaxes are constructed
- - for levels = 0 or 1, 'sigmoid' architecture makes multiple sigmoid models with an ensemble of single-taskers scheme...each model trains to predict '1' for its target class, or zero for a null model constructed from a random sampling from all of the other classes...the size of the random sample is limited to equal the number of images for the target class
- - if levels > 1, 'sigmoid' makes an ensemble of single-taskers for the outer level, and ensembles for the inner levels as well
- - the number of levels required for sigmoid is one larger than for softmax since softmax makes a model from the outer folder that contains the inner folders, while sigmoid makes a model for each of the inner folders
- - with the present implementation, it's not possible to configure different types (sigmoid or softmax) for different levels
- model architecture
- - you may specify your own architecture for the model (which will be the same for every model) by making a 'factory' function that constructs the model, then pass the factory function to the make_models function (refer to examples in custom or repository files)
- repositories
- - you may also load images from an online repository...set images_folder = None when initializing the umbrella network, and initialize_now = False
- - load the images from the repository yourself, and, if you want a multi-level network, organize them into clusters of classes
- - keep track of the labels and class names, but you won't need to pass the labels to the umbrella network...it makes its own labels...for example, if MNIST is organized into odd (0,2,4,6,8) and even (1,3,5,7,9), the labels for both are (0,1,2,3,4). How will you know which is which? Each node in the umbrella network keeps track of its class 'name', which is sometimes simply the folder name or the path to the folder. For a repository, you set the class names when you initialize each node. 
- prediction
- - all folders (classes) must have unique names, and accuracy is determined by comparing the predicted name with the expected name...any repeated names could result in false positives
- - when predicting, to see the names of the predicted classes, set 'verbose=True' when invoking the predict_validation_set or predict_test_set functions. However, then the size of the validation and test sets should be set small, since every image and its predicted class name will be displayed.
- - for all architectures except joint probability, prediction is performed with traversal by maximum...the maximum prediction at the first level selects the subset for prediction at the next level, and that process is repeated to the specified number of levels, then the prediction of the last level is taken
- - for joint probability, both models are flat, and prediction is performed by multiplying their probabilities, then taking the maximum prediction
- - neither 'sigmoid' nor 'softmax' has a single model that predicts every class, unless the architecture is 'softmax' with levels = 0. For sigmoid, if the number of levels is set to reach the bottom level, then a model will be made for every class, with the returned logits being the same as for a softmax over those classes.
- - the logits are returned from the predict_validation_set and predict_test_set functions. The logits only cover every class if the number of levels reaches to the bottom layer, or levels = 0.
- - the umbrella network has special multi-level labels. To view the labels, use the function traverse_validation_labels. All of the labels, even for the training set, are referred to as validation labels. The logits are formed from the last predicted level of the labels.
- accuracy
- - for multi-level models, accuracy is measured with traversal by maximum, then the accuracy of the final model is 1 if correct, 0 otherwise
- saving and loading
- - the functions are save and load
- - images are saved separately from models, otherwise the files are too large (also, cannot serialize entire network because models won't save that way, instead, the program just serializes data from images into one file and saves model files in a folder)
- - for reloading later, should reinitialize network with initialize_now=False, then load
- - the save_just_images and load_just_images functions were designed for reusing the same images with different models, which must have the same type (sigmoid or softmax), but the functions may have been deprecated
- sitemap
- - umbrella.py
- - - required
- - custom
- - - how to load your own images into an umbrella network, with demonstration files from a private repository not yet publicly available
- - repository
- - - how to load images from an online repository into an umbrella network
- - softmax
- - - demonstration of softmax architecture in various contexts
- - sigmoid
- - - demonstration of sigmoid architecture in various contexts
- - etc
- - - demonstration of helpful functions like save, load, and displaying images and labels when predicting validation or test set
- - joint probability network
