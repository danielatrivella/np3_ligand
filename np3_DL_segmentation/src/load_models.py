import models_ME.minkunet as mnet
import models_ME.resnet as resnet

MODELS = []
# ['MinkUNet101', 'MinkUNet14', 'MinkUNet14A', 'MinkUNet14B', 'MinkUNet14C', 'MinkUNet14D', 'MinkUNet18', 'MinkUNet18A', 'MinkUNet18B', 'MinkUNet18D', 'MinkUNet34', 'MinkUNet34A', 'MinkUNet34B', 'MinkUNet34C', 'MinkUNet50', 'MinkUNetBase', 'ResNetBase']

def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])

add_models(mnet)
add_models(resnet)

def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS

def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]
  #
  return NetClass
