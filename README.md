# 23ZCd

## Dependencies
`pip install -r requriements.txt`

## 0_files
All the files (NN dataset, NN, trajectory, etc.) generated during training and control

## 1_sim
To generate the dataset for LSTM training

`python data_generation_dynamic.py`

## 2_four_LSTM
To train one LSTM controller for each module

`python main.py --mode train`

To test

`python main.py --mode test`

## 3_ctrl
### 1_mapping_large_LSTM
To train one LSTM controller for the four-module robot

`python main.py --mode train`

To test

`python main.py --mode test`

### 2_ctrl_large_LSTM
To control the robot to generate [task] motion (including 'spiral', 'obs', and 'straight')

`python ctrl.py --task [task]`

To generate experimental results figures and gif,

`python show.py --task [task]`

`python plot.py --task [task]`

## 4_biLSTM, 5_six_module

Share the same principle wit 2_ctrl_large_LSTM
