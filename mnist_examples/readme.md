# MNIST Example

## QUICK START
### 0. Create Project in [Stadle.ai](https://stadle.ai/login/?next=/)
login and create project

### 1.Download Stadle Client
#### Local Ver
```shell
git clone https://github.com/tie-set/stadle_client

# install environment
./install.sh

# enter environment
cd stadle_client && source ENVCLIENT/bin/activate 
```

#### pip install ver
```shell
pip install stadle-client ((after verb) lacking time to ...)
```


### 2.Upload BaseModel
※ initialize aggregator and copy paste aggregator_ip & port
```shell
pip install -r requirements.txt

python mnist_admin_agent.py
```

### 3.Start ML Server
```shell
python mnist_ml_agent.py
```
