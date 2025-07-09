# teeth_segmentor
HySpark based model for ToothFairy3 challange

run with:
```
docker run --rm -it --gpus all \
  -v /home/jantyc/data:/data \
  teeth 
  python toothfairy_train.py --num_epochs 10 --data_dir '/home/jantyc/data' 

