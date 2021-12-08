# THUSS
Final Project for THU Speech signal digital processing class

## Reproduce on IEMOCAP
Result averaged over 5 runs (only the fine-tuning stage is ran 5 times) with standard deviation:

### Prepare IEMOCAP
Obtain [IEMOCAP](https://sail.usc.edu/iemocap/) from USC
```
python make_16k.py &&
python gen_meta_label.py &&
python generate_labels_sessionwise.py &&
```