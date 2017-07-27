## Prerequisite
- Tensorflow
- Tensorboard
- Mysql server

## Usage
Please modify the database account and password in `.py` files accordingly. 

Before running the following steps, make sure that the `log` directory is empty! Otherwise the graph on TensorBoard will look messy.

```
sudo python mysql_create.py
sudo python svm_db.py
tensorboard --log-dir=log
```