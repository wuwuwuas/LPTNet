import time
import openlpt as lpt 
import sys
import time
import pyopenlpt as lpt 
from demo.sub_modules import Logger

ppp = '0.05'
config_file = rF'./test/ISO_TR_{ppp}ppp/config_python_{ppp}_STB_C2.txt'
log_file = rF'./test/ISO_TR_{ppp}ppp/log_STB_C2.txt'

t_start = time.perf_counter()
sys.stdout = Logger(log_file, sys.stdout)

with lpt.PythonStreamRedirector():
    lpt.run(config_file)
t_end = time.perf_counter()
print(f"Total time {t_end - t_start} seconds")