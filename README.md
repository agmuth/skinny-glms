# skinny-glms
Bare bones IRLS solver designed for speed, rcomposibility, and extensibility.

## Installation
```{code}
    pip install git+https://github.com/agmuth/skinny-glms.git
```

## Examples
```{code}
    examples/
```

## Speed Comparison 

| Family | Link | n | p | skinnyglms_micro_secs | statsmodels_micro_secs | 
|----|----|----|----|----|----|
| Gaussian | Identity | 10 | 0 | 22 | 519 | 
| Binomial | Logit | 10 | 0 | 130 | 510 | 
| Gamma | Log | 10 | 0 | 139 | 1183 | 
| Poisson | Log | 10 | 0 | 142 | 701 | 
| InverseGaussian | Log | 10 | 0 | 137 | 1247 | 
| Gaussian | Identity | 10 | 1 | 22 | 530 | 
| Binomial | Logit | 10 | 1 | 167 | 704 | 
| Gamma | Log | 10 | 1 | 214 | 2115 | 
| Poisson | Log | 10 | 1 | 169 | 880 | 
| InverseGaussian | Log | 10 | 1 | 212 | 3736 | 
| Gaussian | Identity | 100 | 0 | 21 | 534 | 
| Binomial | Logit | 100 | 0 | 148 | 861 | 
| Gamma | Log | 100 | 0 | 154 | 1254 | 
| Poisson | Log | 100 | 0 | 159 | 720 | 
| InverseGaussian | Log | 100 | 0 | 160 | 1310 | 
| Gaussian | Identity | 100 | 1 | 24 | 557 | 
| Binomial | Logit | 100 | 1 | 152 | 895 | 
| Gamma | Log | 100 | 1 | 164 | 1791 | 
| Poisson | Log | 100 | 1 | 166 | 907 | 
| InverseGaussian | Log | 100 | 1 | 198 | 4427 | 
| Gaussian | Identity | 100 | 10 | 32 | 704 | 
| Binomial | Logit | 100 | 10 | 255 | 1162 | 
| Gamma | Log | 100 | 10 | 1700 | 12393 | 
Poisson | Log | 100 | 10 | 218 | 1103
| InverseGaussian | Log | 100 | 10 | 3688 | 17262 | 
| Gaussian | Identity | 1000 | 0 | 29 | 792 | 
| Binomial | Logit | 1000 | 0 | 335 | 1080 | 
| Gamma | Log | 1000 | 0 | 326 | 1981 | 
| Poisson | Log | 1000 | 0 | 333 | 1057 | 
| InverseGaussian | Log | 1000 | 0 | 396 | 2103 | 
| Gaussian | Identity | 1000 | 1 | 35 | 914 | 
| Binomial | Logit | 1000 | 1 | 348 | 1178 | 
| Gamma | Log | 1000 | 1 | 343 | 2661 | 
| Poisson | Log | 1000 | 1 | 369 | 1192 | 
| InverseGaussian | Log | 1000 | 1 | 418 | 3050 | 
| Gaussian | Identity | 1000 | 10 | 87 | 1949 | 
| Binomial | Logit | 1000 | 10 | 930 | 3036 | 
| Gamma | Log | 1000 | 10 | 1141 | 11031 | 
| Poisson | Log | 1000 | 10 | 878 | 3509 | 
| InverseGaussian | Log | 1000 | 10 | 1327 | 14423 | 
| Gaussian | Identity | 1000 | 100 | 752 | 19381 | 
| Binomial | Logit | 1000 | 100 | 5122 | 24817 | 
| Gamma | Log | 1000 | 100 | 12195 | 285360 | 
| Poisson | Log | 1000 | 100 | 5062 | 27484 | 
| InverseGaussian | Log | 1000 | 100 | 87981 | -1 | 
