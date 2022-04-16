import os
import sys
import numpy as np
import math
import random
from PIL import Image, ImageOps, ImageChops
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, add, BatchNormalization, Dropout, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import backend as backend
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import copy
import time

# author Derek James Smith
# derekjsmit@gmail.com

class Joint_Probability_Network(object):

  def __init__(self,images_folder,serial_file,models_folder,target_image_size,channels,valid_ratio,test_ratio):
    self.map = []
    network_type = "softmax"
    levels = 1
    initialize_now = True
    self.specialist = Umbrella_Network(network_type,images_folder,serial_file,models_folder,target_image_size,channels,levels,valid_ratio,test_ratio,initialize_now)
    self.make_map(self.specialist.ROOT)
    initialize_now = False
    levels = 0
    self.generalist = Umbrella_Network(network_type,images_folder,serial_file,models_folder,target_image_size,channels,levels,valid_ratio,test_ratio,initialize_now)

  def make_map(self,root): # for specialist, count how many subclasses found in each class
    folder = root.path
    for filename in sorted(os.listdir(folder)):
      filepath = "{0}/{1}".format(folder,filename)
      if os.path.isdir(filepath):
        self.map.append(0)
    for c in range(0,len(root.children)):
      child = root.children[c]
      self.count_classes(child.path,c)
  
  def count_classes(self,folder,index):
    for filename in sorted(os.listdir(folder)):
      filepath = "{0}/{1}".format(folder,filename)
      if os.path.isdir(filepath):
        self.count_classes(filepath,index)
      else:
        self.map[index] += 1
        break

  def traverse_file_tree(self,folder):
    for filename in sorted(os.listdir(folder)):
      filepath = "{0}/{1}".format(folder,filename)
      print(filepath)
      if os.path.isdir(filepath):
        self.traverse_file_tree(filepath)
  
  def traverse_tree_nodes(self,node):
    print(node.path)
    for child in node.children:
      print(child.path)
      if len(child.children) > 0:
        for c in child.children:
          self.traverse_tree_nodes(c)

  def traverse_training_data(self):
    print("specialist")
    self.specialist.traverse_training_data()
    print("generalist")
    self.generalist.traverse_training_data()

  def make_models(self):
    print("making models for specialist")
    self.specialist.make_models()
    print("making modes for generalist")
    self.generalist.make_models()

  def train_models(self,eps=40,batch=10,aug=False,vrb=True):
    print("training specialist")
    self.specialist.train_models(eps,batch,aug,vrb)
    print("training generalist")
    self.generalist.train_models(eps,batch,aug,vrb)

  def get_training_accuracy(self):
    print("training accuracy")
    print("specialist")
    self.specialist.get_training_accuracy()
    print("generalist")
    self.generalist.get_training_accuracy()

  def accuracy(self,expected,predicted):
    return np.mean([self.ac(expected[j],predicted[j]) for j in range(0,len(expected))])
  
  def ac(self,expected,predicted): # whether top 1 prediction is correct
    return 1 if np.argmax(expected)==np.argmax(predicted) else 0

  def precision_accuracy(self,expected,predicted): # how close is top 1 prediction
    return 1 - np.abs(expected[np.argmax(expected)] - predicted[np.argmax(expected)])

  def precision(self,expected,predicted):
    return np.mean([self.prec(expected[j],predicted[j]) for j in range(0,len(expected))])

  def prec(self,expected,predicted): # all indices tested
    return np.mean([1 - np.abs(expected[j] - predicted[j]) for j in range(0,len(predicted))])

  def softmax_vector(self,y):
    minimum = min(y)
    y -= abs(minimum)
    maximum = max(y)
    y /= maximum
    summation = sum([math.exp(x) for x in y])
    result = [math.exp(x) / summation for x in y]
    return result

  def predict_validation_set(self):
    print("validation accuracy")
    valid_specialist,predicted_specialist = self.predict_set(self.specialist,self.specialist.validation_set);
    valid_general,predicted_general = self.predict_set(self.generalist,self.generalist.validation_set);
    print("specialist")
    es = [list(valid_specialist[j].probability) for j in range(0,len(valid_specialist))]
    print("expected %s" % str(es))
    ps = [list(predicted_specialist[j].probability) for j in range(0,len(predicted_specialist))]
    ps = [self.softmax_vector(ps[j]) for j in range(0,len(ps))]
    print("predicted %s" % str(ps))
    print("accuracy %.3f" % self.accuracy(es,ps))
    #self.specialist.traverse_validation_labels()
    print("generalist")
    expected = [list(valid_general[j].probability) for j in range(0,len(valid_general))]
    print("expected %s" % str(expected))
    pg = [list(predicted_general[j].probability) for j in range(0,len(predicted_general))]
    pg = [self.softmax_vector(pg[j]) for j in range(0,len(pg))]
    print("predicted %s" % str(pg))
    print("accuracy %.3f" % self.accuracy(expected,pg))
    #self.generalist.traverse_validation_labels()
    #print("map")
    #print(str(self.map))
    cumulative = [0]
    for i in range(1,len(self.map)+1):
      cumulative.append(self.map[i-1] + cumulative[i-1])
    #print("cumulative")
    #print(str(cumulative))
    expanded = [[0] * len(pg[0]) for j in range(0,len(ps))]
    #print("expanded specialist")
    #print(str(expanded))
    for i in range(0,len(expanded)):
      for j in range(1,len(cumulative)):
        expanded[i][cumulative[j-1]:cumulative[j]] = [ps[i][j-1]] * len(expanded[i][cumulative[j-1]:cumulative[j]])
    #print("expanded specialist")
    #print(str(expanded))
    predicted = [list(np.array(pg[j]) * np.array(expanded[j])) for j in range(0,len(expanded))]
    predicted = [self.softmax_vector(predicted[j]) for j in range(0,len(predicted))]
    print("Joint probability")
    #print("generalist predictions")
    #print(str(pg))
    #print("divided by expanded specialist predictions")
    #print(str(list(expanded)))
    print("expected %s" % str(expected))
    print("predicted %s" % str(predicted))
    print("accuracy %.3f" % self.accuracy(expected,predicted))

  def predict_set(self,model,someset):
    valid_labels = [node.validation_label for node in someset]
    predicted_labels = model.predict_models(someset)
    return valid_labels,predicted_labels

class Umbrella_Label(object):
  def __init__(self):
    self.probability = []
    self.name = ""
    self.children = []
    self.parent = None

class Umbrella_Node(object):
  def __init__(self,path,parent):
    self.path = path
    if self.path != None:
      self.name = self.path[self.path.rfind("/") + 1 :]
    else:
      self.name = path
    self.parent = parent
    self.children = []
    self.visited = False
    self.count = 0 # number of images contained in child nodes
    self.data = None
    self.validation_label = None
    self.training = True
    self.validation = False
    self.test = False
    self.k = 0
    self.cross_validations = 1
    self.model = None
    self.model_file_path = None
    self.targets = []
    self.nulls = []
    self.trainX = []
    self.trainY = []
    self.probability = 0
    self.temp = False
    self.history = None
    self.training_accuracy = -1
    self.training_loss = -1

class Umbrella_Network(object):

  def instructions(self):
    print("instructions")
    print("network = Umbrella_Network(network_type,images_folder,serial_file,models_folder,target_image_size,channels,levels,valid_ratio,test_ratio,multilabel,initialize_now)\n")

  def __init__(self,network_type,images_folder,serial_file,models_folder,target_image_size,channels,levels,valid_ratio,test_ratio,initialize_now,multilabel=False):
    if network_type==None or serial_file==None or models_folder==None or target_image_size==None or channels==None or levels==None or valid_ratio==None or test_ratio==None or multilabel==None or initialize_now==None:
      print("error - missing constructor parameters")
      self.instructions()
      sys.exit()
    if network_type != 'sigmoid' and network_type != 'softmax':
      print("error - unknown network type, please choose sigmoid or softmax")
      self.instructions()
      sys.exit()
    if images_folder==None:
      print("warning - no source image folder specified\ninstead, you must construct tree manually with repository data")
    elif not os.path.isdir(images_folder):
      print("warning - could not find images folder")
    if not os.path.isdir(models_folder):
      print("warning - could not find models folder")
    if not isinstance(serial_file,str):
      print("error - serial_file should be a string")
      self.instructions()
      sys.exit()
    if not isinstance(target_image_size,int):
      print("error - target image size should be one integer")
      self.instructions()
      sys.exit()
    if not isinstance(channels,int) or not isinstance(levels,int):
      print("error - levels and channels should be integer")
      self.instructions()
      sys.exit()
    if not isinstance(valid_ratio,float) or not isinstance(test_ratio,float):
      print("error - valid_ratio and test_ratio should be floats")
      self.instructions()
      sys.exit()
    if not isinstance(multilabel,bool) or not isinstance(initialize_now,bool):
      print("error - multilabel and initialize_now should be boolean")
      self.instructions()
      sys.exit()
    self.NETWORK_TYPE = network_type
    self.IMAGES_FOLDER = images_folder # read images, if None defaults to repo mode
    self.SERIAL_FILE = serial_file # save images
    self.MODELS_FOLDER = models_folder # save models
    self.TARGET_IMAGE_SIZE = target_image_size # just 1 integer, resize to square
    self.CHANNELS = channels
    self.LEVELS = levels # depth of network
    if self.NETWORK_TYPE=='sigmoid':
      self.LEVELS += 1
    self.validation_set = []
    self.validation_ratio = valid_ratio
    self.test_set = []
    self.test_ratio = test_ratio
    self.training_accuracies = []
    self.training_losses = []
    self.validation_accuracies = []
    self.validation_losses = []
    self.validation_precisions = [] 
    self.epochs = 1
    self.save_model_count = 1 # model file numbers
    self.load_model_count = 1
    self.multilabel = multilabel # not yet written
    if self.NETWORK_TYPE=='softmax':
      self.multilabel = False
    self.ROOT = Umbrella_Node(self.IMAGES_FOLDER,None)
    if self.LEVELS==0: # structureless (flat) network
      self.LEVELS = 1
      self.add_images(self.ROOT,self.ROOT.path)
      self.make_training_validation_test()
    elif initialize_now==True and self.IMAGES_FOLDER != None:
      self.initialize()

  # structureless network functions (just last level flattened out)

  def add_images(self,node,folder):
    added_parent = False
    for filename in sorted(os.listdir(folder)):
      filepath = "{0}/{1}".format(folder,filename)
      if os.path.isdir(filepath):
         self.add_images(node,filepath)
      elif filename != ".DS_Store" and filename != "." and filename != "..":
        if added_parent==False:
          parent = Umbrella_Node(folder,node)
          self.add_umbrella_node(node,parent)
          added_parent = True
        child = Umbrella_Node(filepath,node)
        img = Image.open(filepath)
        img = img.resize((self.TARGET_IMAGE_SIZE,self.TARGET_IMAGE_SIZE))
        data = np.asarray(img).astype(np.float32) / 255
        child.data = data
        self.add_umbrella_node(parent,child)

  # umbrella label functions

  def add_label_node(self,parent,child):
    child.parent = parent
    parent.children.append(child)

  def get_depth_to_root_label(self,node):
    depth = 1
    parent = node.parent
    while parent != None:
      depth += 1
      parent = parent.parent
    return depth

  # umbrella node functions

  def add_umbrella_node(self,node,child):
    node.children.append(child) # parent already added
    if self.image_umbrella_node(child):
      node.count += 1
      # propagate up through all parents
      parent = node.parent
      while parent != None:
        parent.count += 1
        parent = parent.parent

  def image_umbrella_node(self,node):
    if node.path==None:
      return False
    path = node.path.lower()
    if path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png") or path.endswith(".gif") or path.endswith(".tif") or path.endswith(".tiff") or path.endswith(".bmp"):
      return True
    if not os.path.isdir(node.path):
      print("image file not recognized {0}".format(node.path))
    return False

  # repository mode functions (read just data from online repository)

  def set_repo_root(self,name):
    if self.IMAGES_FOLDER != None:
      print("error - adding root in repository mode, but source image folder also specified")
      sys.exit()
    self.ROOT = Umbrella_Node(None,None)
    self.ROOT.name = name
    return self.ROOT

  def add_repo_node(self,parent,name=None):
    if self.IMAGES_FOLDER != None:
      print("error - adding nodes in repository mode, but source image folder also specified")
      sys.exit()
    child = Umbrella_Node(None,parent)
    child.name = name
    self.add_umbrella_node(parent,child)
    return child

  # make leaf nodes containing data from images, or just add a blank node
  # ignores y labels, makes later, so images must all have same label value
  def add_repo_nodes(self,parent,x=None,name=None):
    if self.IMAGES_FOLDER != None:
      print("error - adding nodes in repository mode, but source image folder also specified")
      sys.exit()
    if not isinstance(x,list) and not isinstance(x,np.ndarray):
      print("error - no nodes")
      sys.exit()
    # make images into leaf nodes
    for j in range(0,len(x)):
      child = Umbrella_Node(None,parent)
      child.data = copy.deepcopy(x[j])
      child.name = name
      self.add_umbrella_node(parent,child)
    #for j in range(0,len(parent.children)):
      #parent.trainX.append(parent.children[j]) # trainX has nodes, not images
    # ignore y, must remake

  def init_from_repo(self):
    if self.IMAGES_FOLDER != None:
      print("error - initializing repository mode but source images folder also specified")
      sys.exit()
    self.make_training_validation_test()

  # non-repository mode - traverse folder structure, folders contain images

  def initialize(self):
    self.make_file_tree(self.ROOT)
    self.make_training_validation_test()

  def make_file_tree(self,root):
    folder = root.path
    for filename in sorted(os.listdir(folder)):
      filepath = "{0}/{1}".format(folder,filename)
      if os.path.isdir(filepath):
        child = Umbrella_Node(filepath,root)
        self.add_umbrella_node(root,child)
        self.make_file_tree(child)
      elif filename != ".DS_Store":
        child = Umbrella_Node(filepath,root)
        img = Image.open(filepath)
        img = img.resize((self.TARGET_IMAGE_SIZE,self.TARGET_IMAGE_SIZE))
        data = np.asarray(img).astype(np.float32) / 255
        child.data = data
        self.add_umbrella_node(root,child)

  # both modes

  def make_training_validation_test(self):
    self.make_validation_set(self.validation_ratio,self.test_ratio)
    self.validation_set = []
    self.get_validation_set()
    self.make_validation_labels()
    self.make_training_data()
    self.make_test_set(self.validation_ratio,self.test_ratio)
    self.test_set = []
    self.get_test_set()
    self.trim_validation_set()

  # save functions, could rewrite, presently deletes all models before saving images

  # before save data
  def remove_models(self,node=None):
    if node==None:
      node = self.ROOT
    node.model = None
    for c in node.children:
      self.remove_models(c)

  # save umbrella nodes, including extra attributes other than images
  def save_images(self,path=None):
    if path==None:
      path = self.SERIAL_FILE
    self.remove_models()
    w = open(path,"wb")
    data = self.ROOT
    pickle.dump(data,w)
    w.close()

  # load umbrella nodes
  def load_images(self,path=None):
    if path==None:
      path = self.SERIAL_FILE
    fileptr = open(path,"rb")
    if os.access(path,os.F_OK):
      self.ROOT = pickle.load(fileptr)
      self.make_training_validation_test()
    else:
      print("file not loaded\n")

  # saves less data
  def save_just_images(self,path=None):
    if path==None:
      path = self.SERIAL_FILE
    # breadth first search
    root1 = self.ROOT
    root2 = Umbrella_Node(root1.path,None)
    root3 = root2
    queue1 = []
    queue1.insert(0,root1)
    queue2 = []
    queue2.insert(0,root2)
    while len(queue1) > 0:
      root1 = queue1.pop()
      root2 = queue2.pop()
      root2.data = root1.data
      # visit
      for c in root1.children:
        queue1.insert(0,c)
        child = Umbrella_Node(c.path,root2)
        self.add_umbrella_node(root2,child)
        queue2.insert(0,child)
    w = open(path,"wb")
    data = root3
    pickle.dump(data,w)
    w.close()
  
  # works for different umbrella networks that have same topology
  # load from pre-saved serialized model, but just load image data, not models
  # faster than reloading all images then converting them to data arrays
  def load_just_images(self,path=None):
    if path==None:
      path = self.SERIAL_FILE
    fileptr = open(path,"rb")
    if os.access(path,os.F_OK):
      self.ROOT = pickle.load(fileptr)
      self.make_training_validation_test()
    else:
      print("file not loaded\n")

  # prior to training, mark validation nodes as off limits for training
  def make_validation_set(self,valid_ratio,test_ratio):
    self.reset_validation_set()
    # first make validation set, then make test set a subset of validation set
    self.init_validation_set(valid_ratio + test_ratio)

  def make_test_set(self,valid_ratio,test_ratio):
    # first make validation set, then make test set a subset of validation set
    self.init_test_set(test_ratio / (valid_ratio + test_ratio))

  def reset_validation_set(self,node=None):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      if len(node.children[0].children)==0:# jpg files
        for i in range(0,len(node.children)):
          node.children[i].training = True
          node.children[i].validation = False
          node.children[i].test = False
      else:
        for c in node.children:
          self.reset_validation_set(c)

  def init_validation_set(self,ratio,node=None):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      if len(node.children[0].children)==0:# jpg files
        indices = [j for j in range(0,len(node.children))]
        shuffle(indices)
        index = int(np.round(len(indices) * ratio))
        indices = indices[0:index]
        for i in range(0,len(indices)):
          node.children[indices[i]].training = False
          node.children[indices[i]].validation = True
          node.children[indices[i]].test = False
      else:
        for c in node.children:
          self.init_validation_set(ratio,c)

  def init_test_set(self,ratio,node=None):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      if len(node.children[0].children)==0:# jpg files
        indices = [j for j in range(0,len(node.children)) if node.children[j].validation==True]
        shuffle(indices)
        index = int(np.round(len(indices) * ratio))
        indices = indices[0:index]
        for i in range(0,len(indices)):
          node.children[indices[i]].training = False
          node.children[indices[i]].validation = False
          node.children[indices[i]].test = True
      else:
        for c in node.children:
          self.init_test_set(ratio,c)

  def get_validation_set(self,node=None):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      for c in node.children:
        self.get_validation_set(c)
    elif node.training==False and node.validation==True:# jpg file
      self.validation_set.append(node)

  def get_test_set(self,node=None):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      for c in node.children:
        self.get_test_set(c)
    elif node.training==False and node.validation==False and node.test==True:# jpg file
      self.test_set.append(node)

  def trim_validation_set(self,node=None):
    if node==None:
      node = self.ROOT
      del self.validation_set[:]
    if len(node.children) > 0:
      for c in node.children:
        self.trim_validation_set(c)
    elif node.training==False and node.validation==True:# jpg file
      self.validation_set.append(node)

  # make the correct validation label for an image node
  # the validation labels are stored at the bottom of the umbrella network in the image nodes
  # the validation labels are trees that are only used for the validation set
  def make_validation_label(self,targetnode,label,node=None):
    # depth first search
    # traverse umbrella tree until find target leaf node
    if node==None:
      node = self.ROOT
    elif node==targetnode:
      # found target leaf node, make label by traversing blank label to root, adding ones
      prev = label
      parent = label.parent
      while parent != None:
        if parent.probability != None and len(parent.probability) > 0:
          index = parent.children.index(prev)
          parent.probability[index] = 1
        prev = parent
        parent = parent.parent
    # add empty label of zeros to present node
    label.probability = []
    label.name = node.name
    for c in node.children:
      if len(c.children) > 0: 
        label.probability.append(0)
    # traverse blank label at same time as umbrella tree
    for c in node.children:
      child = Umbrella_Label()
      child.parent = label
      if len(c.children) > 0: 
        self.add_label_node(label,child)
      self.make_validation_label(targetnode,child,c)

  def make_validation_labels(self):
    for v in range(0,len(self.validation_set)):
      node = self.validation_set[v]
      node.validation_label = Umbrella_Label()
      self.make_validation_label(node,node.validation_label)

  def make_training_data(self,node=None,levels = -1,level=1):
    if node==None:
      node = self.ROOT
      levels = self.LEVELS
    if self.NETWORK_TYPE=='sigmoid':
      self.make_sigmoid_training_data(node,levels,level)
    elif self.NETWORK_TYPE=='softmax':
      self.make_softmax_training_data(node,levels,level)
  
  def make_sigmoid_training_data(self,node,levels,level):
    # make targets for all nodes
    self.make_sigmoid_targets(node,levels,level)
    # take sampling from targets of neighbors
    self.make_sigmoid_nulls(node,levels,level)
    # concatenate targets and nulls, then shuffle, then delete targets and nulls
    self.make_sigmoid_training(node,levels,level)

  def make_sigmoid_targets(self,node=None,levels = -1,level=1):
    if level > levels:
      return
    # for sigmoid, root node has no model, jpg nodes have no model, all others have models
    if node.parent == None:
      node.targets = None
    elif len(node.children) > 0:
      node.targets = self.get_image_nodes_for_sigmoid(node)
    else:
      node.targets = None
    for c in node.children:
      self.make_sigmoid_targets(c,levels,level + 1)

  # possible problem for sigmoid - very small training sets, very large null models (large imbalance)
  def make_sigmoid_nulls(self,node=None,levels = -1,level=1):
    if level > levels:
      return
    # for sigmoid, root node has no model, jpg nodes have no model, all others have models
    if node.parent == None:
      node.nulls = None
    elif len(node.children) > 0: # not jpg
      # add same amount of null models as targets
      node.nulls = [] # label 0
      neighbors = len(node.parent.children) - 1
      imgs_per_neighbor = int(float(len(node.targets)) / float(neighbors))
      while imgs_per_neighbor * neighbors < len(node.targets):
        imgs_per_neighbor += 1
      selfindex = node.parent.children.index(node)
      for i in range(0,len(node.parent.children)):
        if i==selfindex:
          continue
        indices = range(0,len(node.parent.children[i].targets))
        k = imgs_per_neighbor
        if k > len(node.parent.children[i].targets):
          k = len(node.parent.children[i].targets)
        samp = random.sample(indices,k)
        node.nulls.extend([node.parent.children[i].targets[j] for j in samp])
    else:
      node.nulls = None
    for c in node.children:
      self.make_sigmoid_nulls(c,levels,level + 1)

  def make_sigmoid_training(self,node=None,levels = -1,level=1):
    if level > levels:
      return
    # for sigmoid, root node has no model, jpg nodes have no model, all others have models
    if node.parent != None and len(node.children) > 0:
      # make shuffled training set with x,y labels both shuffled with stable sort
      node.trainX = np.concatenate( (node.targets,node.nulls), axis=0)
      node.trainY = np.concatenate( ( np.full( (len(node.targets),1), 1), np.full( (len(node.nulls),1), 0) ) , axis=0)
      node.trainY = node.trainY.flatten()
      node.trainX, node.trainY = shuffle(node.trainX,node.trainY)
      del node.targets[:]
      del node.nulls[:]
    for c in node.children:
      self.make_sigmoid_training(c,levels,level + 1)

  # get images nodes for just the first level of folders within the node
  def get_image_nodes_for_sigmoid(self,node):
    if node.parent != None:
      targets = [] # label 1
      nodes = []
      # non-recursive so just iterates outer level folders
      for c in range(0,len(node.children)):
        del nodes[:]
        self.get_image_nodes(node.children[c],nodes)
        targets.extend(nodes)
      return targets
    else:
      return None

  # recursively iterate node and add all images in all subfolders to array
  def get_image_nodes(self,node,nodes):
    if len(node.children) > 0:
      for c in range(0,len(node.children)):
        self.get_image_nodes(node.children[c],nodes)
    elif node.training==True and len(node.children)==0:
      nodes.append(node)

  def make_softmax_training_data(self,node=None,levels = -1,level=1):
    if level > levels:
      return
    # for softmax, root node has model, jpg nodes and nodes above jpgs have no models, all others have models
    if len(node.children) > 0 and len(node.children[0].children) > 0:
      node.trainX, node.trainY = self.get_image_nodes_and_labels_for_softmax(node)
      node.trainX, node.trainY = shuffle(node.trainX,node.trainY)
      node.trainX = np.array(node.trainX)
      node.trainY = to_categorical(node.trainY)
    else:
      node.trainX = None
      node.trainY = None
    for c in node.children:
      self.make_softmax_training_data(c,levels,level + 1)

  # get images nodes for just the first level of folders within the node
  def get_image_nodes_and_labels_for_softmax(self,node=None):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      targets = []
      del targets[:]
      labels = []
      del labels[:]
      # non-recursive so just iterates outer level folders
      for c in range(0,len(node.children)):
        nodes = []
        del nodes[:]
        self.get_image_nodes_for_softmax(node.children[c],nodes)
        targets.extend(nodes)
        labels.extend([c] * len(nodes))
      return targets,labels
    else:
      return None, None

  # recursively iterate node and add all images in all subfolders to array
  def get_image_nodes_for_softmax(self,node,nodes):
    if len(node.children) > 0:
      for c in range(0,len(node.children)):
        self.get_image_nodes_for_softmax(node.children[c],nodes)
    elif node.training==True and len(node.children)==0:
      nodes.append(node)

  def make_models(self,factory=None,verbose=False,node=None,level=1):
    # depth first search
    # does not need breadth first search since increments level in parameters
    if node==None:
      node = self.ROOT
    if level > self.LEVELS:
      return
    if len(node.children) > 0:
      if self.NETWORK_TYPE=='sigmoid':
        if node.parent != None:
          print("making model for {0}".format(node.name))
          if factory==None:
            self.make_sigmoid_model(node)
          else:
            factory(node)
      elif self.NETWORK_TYPE=='softmax':
        if len(node.children) > 0 and not node.children[0].name.endswith("jpg"):
          print("making model for {0}".format(node.name))
          if factory==None:
            self.make_softmax_model(node)
          else:
            factory(node)
    if verbose==True:
      print(node.name)
      node.model.summary()
    for c in node.children:
      if len(c.children) > 0:
        self.make_models(factory,verbose,c,level + 1)

  def make_sigmoid_model(self,node):
    node.model = Sequential()
    node.model.add(Conv2D(64,(5,5),activation='relu',input_shape=(self.TARGET_IMAGE_SIZE,self.TARGET_IMAGE_SIZE,self.CHANNELS)))
    node.model.add(MaxPool2D(pool_size=(4,4)))
    node.model.add(Flatten())
    node.model.add(Dense(16,activation='relu',input_dim=64))
    node.model.add(Dense(1,activation='sigmoid'))
    node.model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

  def make_softmax_model(self,node):
    node.model = Sequential()
    node.model.add(Conv2D(64,(5,5),activation='relu',input_shape=(self.TARGET_IMAGE_SIZE,self.TARGET_IMAGE_SIZE,self.CHANNELS)))
    node.model.add(MaxPool2D(pool_size=(4,4)))
    node.model.add(Flatten())
    node.model.add(Dense(16,activation='relu',input_dim=64))
    node.model.add(Dense(len(node.children),activation='softmax'))
    node.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

  # when training, don't change value of epochs or else will graph inaccurately
  # factory must have form: def train(model,history,x,y,eps,vrb,batch): history=model.fit(x,y,epochs=eps,verbose=vrb,batch_size=batch); return history
  def train_models(self,eps=5,batch=32,aug=False,vrb=False,LR=0,factory=None,node=None,levels = -1,level=1):
    # depth first search
    # does not need breadth first search since increments level in parameters
    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(self.TARGET_IMAGE_SIZE,self.TARGET_IMAGE_SIZE,self.CHANNELS)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )
    reduceLR = ReduceLROnPlateau(monitor='accuracy',factor=LR,patience=5)
    if node==None:
      node = self.ROOT
      levels = self.LEVELS
      self.epochs = eps
    if level > levels:
      return
    if len(node.children) > 0 and node.model != None:
      if vrb==True:
        print("training model for {0}".format(node.name))
        print("trainX {0}".format(len(node.trainX)))
        print("trainY {0}".format(len(node.trainY)))
      X = np.array([n.data for n in node.trainX])
      Y = np.array(node.trainY)
      if aug==True:
        for i in range(0,len(X)):
          imgdata = X[i]
          img = Image.fromarray(np.uint8(imgdata * 255))
          rnd = random.randint(1,4)
          if rnd==1:
            imgs = data_augmentation(np.reshape(img,(1,self.TARGET_IMAGE_SIZE,self.TARGET_IMAGE_SIZE,self.CHANNELS)))
            img = imgs[0]
            imgdata = np.asarray(img).astype(np.float32) / 255
          elif rnd == 2:
            deg = random.randint(1,45)
            exponent = random.randint(1,2)
            sign = (-1)**(exponent)
            deg = deg * sign
            rotate = img.rotate(deg)
            rotate = np.asarray(rotate).astype(np.float32) / 255
            imgdata = rotate
          elif rnd==3:
            flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip = np.asarray(flip).astype(np.float32) / 255
            imgdata = flip
          elif rnd==4:
            dir = random.randint(1,4)
            deg = random.randint(1,10)
            if dir==1:
              shift = ImageChops.offset(img,deg,0)
            elif dir==2:
              shift = ImageChops.offset(img,-deg,0)
            elif dir==3:
              shift = ImageChops.offset(img,0,deg)
            elif dir==4:
              shift = ImageChops.offset(img,0,-deg)
            shift = np.asarray(shift).astype(np.float32) / 255
            imgdata = shift
          X[i] = imgdata
      if self.NETWORK_TYPE=='sigmoid':
        if factory==None:
          if LR > 0:
            node.history = node.model.fit(X,Y,epochs=eps,verbose=vrb,batch_size=batch,callbacks=[reduceLR])
          else:
            node.history = node.model.fit(X,Y,epochs=eps,verbose=vrb,batch_size=batch)
        else:
            node.history = factory(node.model,node.history,X,Y,eps,vrb,batch)
      elif self.NETWORK_TYPE=='softmax':
        if factory==None:
          if LR > 0:
            node.history = node.model.fit(X,Y,epochs=eps,verbose=vrb,batch_size=batch,callbacks=[reduceLR])
          else:
            node.history = node.model.fit(X,Y,epochs=eps,verbose=vrb,batch_size=batch)
        else:
          node.history = factory(node.model,node.history,X,Y,eps,vrb,batch)
      node.training_accuracy = node.history.history['accuracy'][len(node.history.history['accuracy']) - 1]
      node.training_loss = node.history.history['loss'][len(node.history.history['loss']) - 1]
    for c in node.children:
      self.train_models(eps,batch,aug,vrb,LR,factory,c,levels,level + 1)

  def reset_model_counts(self):
    self.save_model_count = 1
    self.load_model_count = 1

  def save_models(self,path=None,node=None,levels = -1,level=0):
    #breadth first search
    if node==None:
      node = self.ROOT
      path = self.MODELS_FOLDER
      levels = self.LEVELS
    if level==0:
      self.reset_model_counts()
    levelcount = 0
    queue = []
    del queue[:]
    queue.insert(0,self.ROOT)
    while len(queue) > 0:
      node = queue.pop()
      # visit node
      if len(node.children) > 0 and node.model != None:
        print("{0}".format(node.name))
        model_file_path = "{0}/model{1}".format(path,self.save_model_count) 
        node.model.save(model_file_path)# SavedModel format
        node.model_file_path = model_file_path
        self.save_model_count += 1
      for c in range(0,len(node.children)):
        if len(node.children[c].children) > 0:
          queue.insert(0,node.children[c])

  def load_models(self,path=None,node=None,levels = -1,level=0):
    #breadth first search
    if node==None:
      node = self.ROOT
      path = self.MODELS_FOLDER
      levels = self.LEVELS
    if level==0:
      self.reset_model_counts()
    levelcount = 0
    queue = []
    del queue[:]
    queue.insert(0,self.ROOT)
    while len(queue) > 0:
      node = queue.pop()
      # visit node
      if len(node.children) > 0 and (node.parent != None or self.NETWORK_TYPE=='softmax'):
        model_file_path = "{0}/model{1}".format(path,self.load_model_count)
        if os.path.exists(model_file_path):
          node.model = load_model(model_file_path) # SavedModel format
          print("model loaded into {0}".format(node.name))
        else:
          print("could not load model for {0}".format(node.name))
        self.load_model_count += 1
      for c in range(0,len(node.children)):
        if len(node.children[c].children) > 0:
          queue.insert(0,node.children[c])

  # returns a label tree of all relevant predictions in umbrella tree
  # for one image, predict with every model that has a 1 probability in its parent, or root for parent
  # possibility of multiple predictions for one image
  # for validation, just evaluate true positives and false negatives
  # for returning a single prediction, traverse predicted label maximums since it's in a tree form
  def predict_models(self,someset):
    if self.NETWORK_TYPE=='sigmoid':
      return self.predict_sigmoids(someset)
    elif self.NETWORK_TYPE=='softmax':
      return self.predict_softmaxes(someset)

  # batch prediction (predict one sample at a time through entire tree was slow)
  def predict_sigmoids(self,someset):
    # breadth first search
    queue = []
    del queue[:]
    # traverse umbrella tree, visiting each model once
    queue.insert(0,self.ROOT)
    while len(queue) > 0:
      node = queue.pop()
      # visit node
      node.probability = []
      if node.model != None and node.parent != None and len(node.children) > 0:
        # predict every node in validation set or test set
        # predict all images with all models, unless node has no model (root, leaves)
        # set is a list, not a tree, do not need to traverse set
        # sigmoid predicts a float
        try:
          data = np.array([setnode.data for setnode in someset])
          predictions = node.model.predict(data)
          predictions = [p.flatten() for p in predictions]
          predictions = [float(p) for p in predictions]
          node.probability = predictions
        except Exception as exc:
          print("exception in predict sigmoids")
          print(str(exc))
          sys.exit()
      for c in range(0,len(node.children)):
        if len(node.children[c].children) > 0:
          queue.insert(0,node.children[c])
    # traverse results, make predicted labels
    predicted_labels = []
    for i in range(0,len(someset)):
      predicted_label = Umbrella_Label()
      self.predict_sigmoid_label(self.ROOT,predicted_label,i)
      predicted_labels.append(predicted_label)
    return predicted_labels

  # batch prediction
  def predict_softmaxes(self,someset):
    #breadth first search
    queue = []
    del queue[:]
    queue.insert(0,self.ROOT)
    while len(queue) > 0:
      node = queue.pop()
      # visit node
      if node.model != None and len(node.children) > 0:
        node.probability = []
        # for batch prediction, predict every image with all models
        # predict all images in validation set or test set
        data = np.array([setnode.data for setnode in someset])
        predictions = node.model.predict(data)
        predictions = [np.hstack(p) for p in predictions]
        node.probability = predictions
      for c in range(0,len(node.children)):
        if len(node.children[c].children) > 0:
          queue.insert(0,node.children[c])
    predicted_labels = []
    for i in range(0,len(someset)):
      predicted_label = Umbrella_Label()
      self.predict_softmax_label(self.ROOT,predicted_label,i)
      predicted_labels.append(predicted_label)
    return predicted_labels

  # sigmoid umbrella nodes have binary probability, label nodes have list probabilities, must convert
  def predict_sigmoid_label(self,node,label,index):
    # depth first search
    # traverse umbrella tree and label tree at same time
    # does not need breadth first search since both are trees and traverses all levels
    label.name = node.name
    label.probability = []
    for c in node.children:
      if len(c.children) > 0 and len(c.probability) > index:
        try:
            label.probability.append(c.probability[index])
        except Exception as exc:
          print("error in predict sigmoid label for index {0}".format(index))
          print("probability label")
          print(c.probability)
          print(str(exc))
          sys.exit()
      #else:
        #label.probability.append(0)
    for c in node.children:
      if len(c.children) > 0:
        child = Umbrella_Label()
        self.add_label_node(label,child)
    for c in range(0,len(node.children)):
      # label tree does not have lowest level
      if len(node.children[c].children) > 0:
        self.predict_sigmoid_label(node.children[c],label.children[c],index)

  # softmax umbrella nodes should already have list probabilities, same as labels
  def predict_softmax_label(self,node,label,index):
    # depth first search
    label.name = node.name
    if not isinstance(node.probability,list) and not isinstance(node.probability,np.ndarray):
      label.probability = []
    else:
      try:
        if len(node.probability) > 0 and len(node.probability) > index:
          label.probability = node.probability[index]
        else:
          label.probability = []
      except Exception as exc:
        print("error in predict softmax label for index {0}".format(index))
        print("probability label")
        print(node.probability)
        print(str(exc))
        sys.exit()
    for c in node.children:
      if len(c.children) > 0:
        child = Umbrella_Label()
        self.add_label_node(label,child)
    for c in range(0,len(node.children)):
      # label tree does not have lowest level
      if len(node.children[c].children) > 0:
        self.predict_softmax_label(node.children[c],label.children[c],index)

  # traverse predicted label and print out predictions at each level
  def get_prediction_from_label(self,label,verbose=True):
    # depth first search of argmaxes only
    # does not need breadth first search
    max = -1
    maxindex = -1
    maxname = ""
    if len(label.children) < 1 or len(label.probability) < 1:
      return
    for c in range(0,len(label.probability)):
      if label.probability[c] > max:
        max = label.probability[c]
        maxname = label.children[c].name
        maxindex = c
        self.temp = maxname
    if max < 0 or maxindex < 0:
      return 
    if verbose==True:
      print("{0} {1}".format(maxname,max))
    for c in range(0,len(label.children)):
      if c==maxindex:
        self.get_prediction_from_label(label.children[c],verbose)

  def predict_validation_set(self,verbose=False):
    self.predict_set(self.validation_set,verbose)
  
  def predict_test_set(self,verbose=False):
    self.predict_set(self.test_set,verbose)
  
  # print average network loss, accuracy, and precision for all samples in someset
  def predict_set(self,someset,verbose=False):
    valid_labels = [node.validation_label for node in someset]
    predicted_labels = self.predict_models(someset)
    accuracies = []
    print(len(someset))
    valid_name = "x"
    predicted_name = "y"
    for i in range(0,len(someset)):
      if verbose==True:
        data = someset[i].data
        plt.imshow(data)
        plt.show()
      if verbose==True:
        print("\nvalid")
      self.get_prediction_from_label(valid_labels[i],verbose)
      valid_name = self.temp
      if verbose==True:
        self.traverse_validation_label(valid_labels[i])
      if verbose==True:
        print("\npredicted")
      self.get_prediction_from_label(predicted_labels[i],verbose)
      predicted_name = self.temp
      if verbose==True:
        self.traverse_validation_label(predicted_labels[i])
      if valid_name==predicted_name:
        accuracies.append(1)
      else:
        accuracies.append(0)
    average_validation_accuracy = np.mean(accuracies)
    if someset==self.validation_set:
      self.validation_accuracies.append(average_validation_accuracy)
      print("")
      print("average validation accuracy %f" % (average_validation_accuracy))
    elif someset==self.test_set:
      print("")
      print("average test accuracy %f" % (average_validation_accuracy))

  # find top n results by flattening tree-shaped labels into 1-d lists
  # option for strict results, only if above the baseline for each sublist

  def predict_validation_top5(self,n=5,verbose=False,baseline=False):
    self.predict_set_top5(self.validation_set,n,verbose,baseline)
  
  def predict_test_top5(self,n=5,verbose=False,baseline=False):
    self.predict_set_top5(self.test_set,n,verbose,baseline)
  
  def predict_set_top5(self,someset,n=5,verbose=False,baseline=False):
    valid_labels = [node.validation_label for node in someset]
    predicted_labels = self.predict_models(someset)
    self.flatten_labels(valid_labels,baseline)
    self.flatten_labels(predicted_labels,baseline)
    accuracy = 0
    for i in range(0,len(valid_labels)):
      index = np.argmax(valid_labels[i])
      length = n
      if n > len(predicted_labels[i]):
        length = len(predicted_labels[i])
      for j in range(0,length):
        maximum = np.argmax(predicted_labels[i])
        # cannot test baseline here since probabilities are joint and have been weighted
        if maximum==index and maximum > 0:
          accuracy += 1
          break
        predicted_labels[i][maximum] = 0
    accuracy = accuracy / len(valid_labels)
    print("top %d accuracy %.2f" % (n,accuracy))
    if verbose==True:
      for i in range(0,len(valid_labels)):
        print("valid {0} predicted {1}".format(valid_labels[i],predicted_labels[i]))

  def flatten_labels(self,labels,baseline=False):
    for j in range(0,len(labels)):
      result = []
      self.flatten_label(labels[j],result,baseline)
      labels[j] = result

  def flatten_label(self,label,result,p=0,level=1,baseline=False):
    # depth first search
    if (isinstance(label.probability,list) or isinstance(label.probability,np.ndarray)) and len(label.probability) > 0:
      for index in range(0,len(label.probability)):
        value = label.probability[index]
        if baseline==True:
          if value < 1.0 / len(label.probability):
            value = 0
        value = value / level
        value += p
        if isinstance(label.children,list) and len(label.children) > 0:
          self.flatten_label(label.children[index],result,value,level + 1,baseline)
        else:
          result.append(value)
    else:
      result.append(p)

  # merge with training set before predicting test set
  def merge_validation_set(self):
    for node in self.validation_set:
      node.training = True
      node.validation = False
      node.Test = False
    del self.validation_set[:]
    self.make_training_data()

  # helpful functions for visualizing the network

  def traverse(self,node=None,indentation = -1):
    # depth first search
    # should not need breadth first search since indentation incremented in parameters
    if node==None:
      node = self.ROOT
      indentation = 0
    if len(node.children) > 0:
      print("{0}{1} {2}".format(" " * indentation,node.name,node.count))
    for c in node.children:
      self.traverse(c,indentation + 3)

  def traverse_validation_label(self,label,indentation=0):
    print("{0}{1} {2}".format(" " * indentation,label.name,label.probability))
    for c in label.children:
      self.traverse_validation_label(c,indentation+2)

  def traverse_validation_labels(self,n=None):
    if n==None:
      n = len(self.validation_set)
    for v in range(0,n):
      self.traverse_validation_label(self.validation_set[v].validation_label)

  def traverse_training_data(self,node=None,indentation=0):
    if node==None:
      node = self.ROOT
    if len(node.children) > 0:
      print("{0}{1} trainX {2} {3} trainY {4} {5}".format(" " * indentation,node.name,len(node.trainX),np.array(node.trainX).shape,len(node.trainY),np.array(node.trainY).shape))
    for c in node.children:
      self.traverse_training_data(c,indentation + 3)
  
  def get_training_accuracy(self,node=None):
    # breadth first search
    if node==None:
      node = self.ROOT
    accuracies = []
    losses = []
    queue = []
    del queue[:]
    queue.insert(0,self.ROOT)
    while len(queue) > 0:
      node = queue.pop()
      # visit node
      if len(node.children) > 0 and node.model != None:
        if node.training_accuracy >= 0:
          accuracies.append(node.training_accuracy)
          losses.append(node.training_loss)
      for c in range(0,len(node.children)):
        if len(node.children[c].children) > 0:
          queue.insert(0,node.children[c])
    accuracy = np.mean(accuracies)
    loss = np.mean(losses)
    self.training_accuracies.append(accuracy)
    self.training_losses.append(loss)
    print("average training accuracy %f" % (accuracy))
    print("average training loss %f" % (loss))
