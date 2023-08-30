import os
import sys
from datetime import datetime
import logging

log_file=f"{datetime.now().strftime('%m_%Y_%d_%H_%M_%S')}.log"

log_path=os.path.join(os.getcwd(),"logs",log_file)
os.makedirs(log_path,exist_ok=True)
log_file_name=os.path.join(log_path,log_file)

logging.basicConfig(filename=log_file_name,level=logging.INFO)

### 
if __name__=="__main__":
    logging.info("logs have created")