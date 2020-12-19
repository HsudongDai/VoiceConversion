# G2GAN based on StarGAN-VC
This repository is a model which conducts voice conversion by G2GAN. The model is based on previous StarGAN-VC job but adds an extra Generator with new loss function. The general structure and data preparation are inherited from previous repo.

## [StarGAN-VC](https://github.com/hujinsen/pytorch-StarGAN-VC)

This is a pytorch implementation of the paper: [StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks](https://arxiv.org/abs/1806.02169).

**The converted voice examples are in *samples* directory**



## [Dependencies](https://github.com/hujinsen/pytorch-StarGAN-VC)
- Python 3.6 
- pytorch 1.0
- librosa 
- pyworld 
- tensorboardX
- scikit-learn


## [Usage](https://github.com/hujinsen/pytorch-StarGAN-VC)

### Download dataset

Download the vcc 2016 dataset to the current directory 

```
python download.py 
```

The downloaded zip files are extracted to `./data/vcc2016_training` and `./data/evaluation_all`.

1. **training set:** In the paper, the author choose **four speakers** from `./data/vcc2016_training`. So we  move the corresponding folder(eg. SF1,SF2,TM1,TM2 ) to `./data/speakers`.
2. **testing set** In the paper, the author choose **four speakers** from `./data/evaluation_all`. So we  move the corresponding folder(eg. SF1,SF2,TM1,TM2 ) to `./data/speakers_test`.

The data directory now looks like this:

```
data
├── speakers  (training set)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── speakers_test (testing set)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── vcc2016_training (vcc 2016 training set)
│   ├── ...
├── evaluation_all (vcc 2016 evaluation set, we use it as testing set)
│   ├── ...
```

### Preprocess

Extract features (mcep, f0, ap) from each speech clip.  The features are stored as npy files. We also calculate the statistical characteristics for each speaker.

```
python preprocess.py
```

This process may take minutes !


### Train

```
python main.py
```



### Convert



```
python main.py --mode test --test_iters 200000 --src_speaker TM1 --trg_speaker "['TM1','SF1']"
```


## [Network structure](https://github.com/hujinsen/pytorch-StarGAN-VC)

![Snip20181102_2](https://github.com/hujinsen/StarGAN-Voice-Conversion/raw/master/imgs/Snip20181102_2.png)



 Note: Our implementation follows the original paper’s network structure, while [pytorch StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion) use StarGAN's network.Both can generate good audio quality. 

## [Reference](https://github.com/hujinsen/pytorch-StarGAN-VC)
[tensorflow StarGAN-VC code](https://github.com/hujinsen/StarGAN-Voice-Conversion)

[StarGAN code](https://github.com/taki0112/StarGAN-Tensorflow)

[CycleGAN-VC code](https://github.com/leimao/Voice_Converter_CycleGAN)


[pytorch-StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN-VC paper](https://arxiv.org/abs/1806.02169)

[StarGAN paper](https://arxiv.org/abs/1806.02169)

[CycleGAN paper](https://arxiv.org/abs/1703.10593v4)

---

If you feel this repo is good, please  **star**  ! 

Your encouragement is my biggest motivation!
