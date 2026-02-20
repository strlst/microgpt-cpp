# microgpt-cpp

This project is a C++ recreation of karpathy's `microgpt.py`, in order to gain an intuition into the process of implementing forward and backward passes of a GPT-like language model, evaluate performance and perhaps most importantly, to have some fun.

Original Python implementation taken from [here](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (made by Andrej Karpathy).

For file downloads, the excellent and simple single-file header-only library [provided by yhirose](https://github.com/yhirose/cpp-httplib/) is used.

## Sample Output

```
Successfully opened file "names.txt"
Initialized vocabulary of size 27
Initialized weights [layer0_attn_wk layer0_attn_wo layer0_attn_wq layer0_attn_wv layer0_mlp_fc1 layer0_mlp_fc2 lm_head wpe wte ] with normdist(mean=0, std_dev=0.08)
Created model(n_embed=16, n_head=4, n_layer=1, head_dim=4)
Training with num_steps=1000
step    0 / 1000 | Loss 3.27773
step    1 / 1000 | Loss 3.29826
step    2 / 1000 | Loss 3.22314
step    3 / 1000 | Loss 3.30184
...
step  997 / 1000 | Loss 1.61079
step  998 / 1000 | Loss 2.00558
step  999 / 1000 | Loss 1.749
Inferring 30 samples with temperature 0.5
sample: alaya
sample: barnen
sample: kiia
sample: kasa
sample: arian
sample: karana
sample: jayla
sample: berenr
sample: joria
sample: aloely
sample: malie
sample: alan
sample: alien
sample: apie
sample: amalena
sample: ailan
sample: cana
sample: jana
sample: kalie
sample: alila
sample: malila
sample: adera
sample: reria
sample: arela
sample: alaey
sample: mera
sample: ajielee
sample: kalile
sample: zeyly
sample: yelenn
```