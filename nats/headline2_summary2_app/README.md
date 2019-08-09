
## Model Framework
<p align="left">
  <img src="figure/multitask_transfer.png" width="500" title="The Model" alt="Cannot Access">
</p>

## Results

| DATASET  | MODEL | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| NEWSROOM-SUMMARY | C10110 MULTI-TASK | 39.85 | 28.37 | 36.91 |
| NEWSROOM-HEADLINE | C10110 MULTI-TASK | 28.31 | 13.40 | 26.64 |
| CNN/DM  | C10110 TRANSFER | 35.55 | 15.19 | 33.00 |
| CNN/DM  | C10111 TRANSFER | 38.49 | 16.78 | 35.68 |
| BYTECUP | C10110 TRANSFER | 40.92 | 24.51 | 38.01 |

## How to use?

#### Step 1
Download pretrain model from https://drive.google.com/open?id=1A7ODPpermwIHeRrnqvalT5zpr4BCTBi9
#### Step 2
Create a folder such as ```data```.
#### Step 3
Set app_data_dir as the above directory.
Run the model.
#### Step 4
In the above directory, create the input ```something_in.json```.
You can create any many input files as you want.
The program will generate ```something_out.json```, which contains the summaries and headlines and automatically delete the ```something_in.json```.

The format of each input file:
```{'content': your article content here}```


