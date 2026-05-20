import sys, os
TRAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
DEPLOY_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../deploy")
sys.path.insert(0, TRAIN_PATH)
