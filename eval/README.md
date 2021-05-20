# Evaluation 

Each challenge has its own evaluation procedure. Please follow instructions for each challenge below. 

All evaluation code depends on my personal toolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox. Please install the toolbox:

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

*2. Install dependency for the toolbox.*
~~~shell
cd Xinshuo_PyToolbox
pip install -r requirements.txt
~~~

## <a href="http://www.aiodrive.org/forecasting.html">Trajectory Forecasting</a>

To run our trajectory forecasting evaluation code, you will first need to prepare the data (result file and ground truth file) in the following path:

```
$ ../data/traj_val_anno.json
$ ../data/traj_val.json
```

Then, you can run the following code for evaluation. Numbers will be printed out and also stored in ../data/eval_traj_log.txt

```
$ python3 traj_pred.py
```

For more details about how to prepare the data (result and ground truth file), you can find instructions <a href="http://www.aiodrive.org/forecasting.html">here</a>. 


