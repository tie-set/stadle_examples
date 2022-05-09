

##　1箇所目2箇所目はfor loopの前
```
初期化
stadle_client = BasicClient(config_file=client_config_path)

modelをstadleにsetする
stadle_client.set_bm_obj(model)
```


なんepochでアグリゲーションするか



```
for epoch in range(num_epochs):
    if (epoch % 2 == 0):
        # Don't send model at beginning of training
        if (epoch != 0):
            stadle_client.send_trained_model(model)
```



### 受け取るまでまち
```
state_dict = stadle_client.wait_for_sg_model().state_dict()
```

### 受け取って
```shell
model.load_state_dict(state_dict)
```



``
stadle_client.disconnect()
``
