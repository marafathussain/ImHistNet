# ImHistNet: Learnable Image Histogram-based DNN

This [paper](http://ece.ubc.ca/~bisicl/papers/mahmiccai19.pdf) is proposed in MICCAI 2019. It is implemented in Caffe. The deatiled instructions of installing Caffe can be found [here](http://caffe.berkeleyvision.org/installation.html). 

## Data preparation
LMDB data format is used in this work. To generate this, there should different training, validation and testing data listing text files which should list the images as "ImageName" "CorrespondingLabel":
```
- train_list.txt
   image_000001.jpg 0
   image_000002.jpg 1
   image_000003.jpg 1
   .......
```
To generate the LMDB files, the following command should be used. You should change the paths inside this file according to yours.

```
>> source create_lmdb.sh
```

Once the lmdb files are created, the following command will generate the mean file. 
```
>> source make_meanfile.sh
```

## Citations
If you find this work useful, please cite one or both of the following papers:
```
@article{hussain2021learnable,
  title={Learnable Image Histograms-based Deep Radiomics for Renal Cell Carcinoma Grading and Staging},
  author={Hussain, Mohammad Arafat and Hamarneh, Ghassan and Garbi, Rafeef},
  journal={Computerized Medical Imaging and Graphics},
  pages={101924},
  year={2021},
  publisher={Elsevier}
}
```

```
@inproceedings{hussain2019imhistnet,
  title={ImHistNet: Learnable Image Histogram Based DNN with Application to Noninvasive Determination of Carcinoma Grades in CT Scans},
  author={Hussain, Mohammad Arafat and Hamarneh, Ghassan and Garbi, Rafeef},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={1-9},
  year={2019},
  organization={Springer}
}
```

```
@inproceedings{hussain2019renal,
  title={Renal Cell Carcinoma Staging with Learnable Image Histogram-based Deep Neural Network},
  author={Hussain, Mohammad Arafat and Hamarneh, Ghassan and Garbi, Rafeef},
  booktitle={International Workshop on Machine Learning in Medical Imaging (MLMI)},
  pages={1-8},
  year={2019},
  organization={Springer}
}
```
