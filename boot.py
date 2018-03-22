import sys, os, code

DL_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DL_BASE_DIR + r"/train/")
os.environ["DL_BASE_DIR"] = DL_BASE_DIR
code.interact(local = locals())
