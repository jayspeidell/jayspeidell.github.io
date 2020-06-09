---
title: "Galaxy Zoo Challenge"
excerpt: "This is an image classification project that I completed for my independent study at Old Dominion University. The dataset was obtained from Kaggle's Galaxy Zoo Challenge. I chose astronomy datasets for my independent study because I enjoy learning about the topic. <br/><img src='/images/galaxy-zoo/header.png'>"
collection: portfolio
---


<img src="/images/galaxy-zoo/header.png" style="width:100%" />
# I. Definition

This is an image classification project that I completed for my independent study at Old Dominion University. I'm interested in astronomy and chose to do Kaggle's [Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge). 

# Set up environment.

{::comment}
```python
from google.colab import drive
import os
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Javascript, display, Markdown, clear_output

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


!pip install Augmentor
import Augmentor


import pickle

# Convenient Pickle wrappers
def save(obj, file):
    pickle.dump(obj, open(SAVE + file ,'wb')) # SAVE is global variable for directory

def load(file):
    return pickle.load(open(SAVE + file, 'rb'))

def RMSE(pred, truth):
    # Element-wise RMSE score.
    return np.sqrt( np.mean( np.square( np.array(pred).flatten() - np.array(truth).flatten() )))

RANDOM = 42
random.seed(RANDOM)

```
{:/comment}


{::comment}
```python
%%capture

display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

# Mount Google Drive and load Kaggle API key into environment.
# You'll have to customize this step for yourself.
drive.mount('/content/gdrive')
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/School/CS497/kaggle/"

# The data is directories of jpg images. A lot of them.
DATA = "/content/images_training_rev1/"
!mkdir /content/processed_64x64
!mkdir /content/processed_128x128
DIR_64 = '/content/processed_64x64/'
DIR_128 = '/content/processed_128x128/'
!mkdir /content/processed_64x64_WIDE
!mkdir /content/processed_128x128_WIDE
DIR_64_WIDE = '/content/processed_64x64_WIDE/'
DIR_128_WIDE = '/content/processed_128x128_WIDE/'



# Directory for saving models
SAVE = "/content/gdrive/My Drive/School/CS497/GalaxyZoo/"
FIGS = "/content/gdrive/My Drive/School/CS497/GalaxyZoo/figs/"
```
{:/comment}
### Download and extract data.

{::comment}
```python
%%capture

# Configure Kaggle to work with Colab and download Galaxy Zoo dataset.
# https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge
# Original paper: https://arxiv.org/abs/1308.3496
# If you are running this in your own, don't forget you have to log into the
# website and manually accept the terms of use for the data.
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!kaggle -v
!kaggle competitions download galaxy-zoo-the-galaxy-challenge

# Unpack training and validation datasets
!unzip /content/galaxy-zoo-the-galaxy-challenge.zip
!rm /content/galaxy-zoo-the-galaxy-challenge.zip
!unzip /content/images_training_rev1.zip
!unzip /content/training_solutions_rev1.zip

```
{:/comment}
#### Load data into DataFrame


```python
images = [f for f in os.listdir(DATA) if os.path.isfile(os.path.join(DATA, f))]
print("There are " + '{:,}'.format(len(images)) + " images in the dataset.")
labels = pd.read_csv('training_solutions_rev1.csv')
labels.GalaxyID = labels.GalaxyID.apply(lambda id: str(int(id)) + '.jpg')
save(labels, 'labels.p')
print("There are " + '{:,}'.format(labels.shape[0]) + " truth values.")
print("There are " + '{:,}'.format(labels.shape[1]-1) + " categories for classification.")
desc = ['Smooth','Featured or disc','Star or artifact','Edge on','Not edge on','Bar through center','No bar','Spiral','No Spiral','No bulge','Just noticeable bulge','Obvious bulge','Dominant bulge','Odd Feature','No Odd Feature','Completely round','In between','Cigar shaped','Ring (Oddity)','Lens or arc (Oddity)','Disturbed (Oddity)','Irregular (Oddity)','Other (Oddity)','Merger (Oddity)','Dust lane (Oddity)','Rounded bulge','Boxy bulge','No bulge','Tightly wound arms','Medium wound arms','Loose wound arms','1 Spiral Arm','2 Spiral Arms','3 Spiral Arms','4 Spiral Arms','More than four Spiral Arms',"Can't tell"]
```

    There are 61,578 images in the dataset.
    There are 61,578 truth values.
    There are 37 categories for classification.


### Utility functions

All of the utility or feature extraction functions are moved up into this cell for convenience.



```python
def average_color(pic):
    '''
    pic is a 4 dimensional array where d0 is the index, d1 and d2 are the
    x and y values of the image and d3 is the values of the pixels (R,G,B
    in this dataset but it will scale to anything).

    I'm taking the mean value of pixels in each image by summing across
    d1 and d2 then dividing by the total number of pixels.
    '''
    return np.sum(pic, axis=(1,2)) / (pic.shape[1] * pic.shape[2])


def center_pixel(pic):
    return pic[:,int(pic.shape[1] / 2),int(pic.shape[2] / 2),:]


def image_generator(pics, path=DATA, batch_size=30, rotate=False, size=100, save=False,
                    retrieve=False):
    '''
    Generate batches of numpy arrays from a list of image filenames.
    DATA is a global variable with the path to the image folder.  

    Output array has 4 dimensions in this order:
        - Index of pictures in this batch
        - x dimension of pixels in individual image
        - y dimension of pixels in individual image
        - Color depth (R,G,B in this project, but it can scale)
    '''
    l = len(pics)
    batches = int(l/batch_size)
    leftover = l % batch_size
    for batch in range(batches):
        start = batch * batch_size
        this_batch = pics[start:start+batch_size]
        if rotate:
            yield np.array([
                            scipy.misc.imresize(
                                scipy.ndimage.rotate(
                                plt.imread(path + pic, format='jpg'),
                                reshape=False,
                                angle=random.randint(0,360, random_seed=RANDOM)
                                ),
                                size=size
                            )

                            for pic in this_batch])
        else:
            yield np.array([ plt.imread(path + pic, format='jpg') for pic in this_batch])
    start = batches * batch_size
    this_batch = pics[start:start+leftover]
    yield np.array([ plt.imread(path + pic, format='jpg') for pic in this_batch])

```

# Exploratory Data Analysis

## Labels

This dataset was hand-labeled for the [Galaxy Zoo 2](https://arxiv.org/abs/1308.3496) project, using an online [crowdsourcing tool](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo). The data was downloaded from Kaggle's [Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) competition via the Kaggle CLI tool. The original images in the Galazy Zoo dataset contain much more information than these jpg images, and that information was removed to focus the challenge around modelling the image analysis process of the humans viewing these images.

The labels represents a decision tree labeled with 37 classes of the format "ClassA.B", where A represents the level in the decision tree (from 1 to 11) and B represents the choices at the given level. This is different than a typical classification problem because each class represents an attribute, and most images will contain multiple attributes. The values represent the confidence of the crowded that a given answer is correct, from zero to one.

The decision process traverses forwards and backwards through a list of questions, and each questions and answer combination is represented as a class in the truth labels. The decision tree must always terminate with an END class, which includes 1.3, 6.2, and 7.x.  

The decision tree described in the paper is as follows:

1. Is the galaxy simply smooth and rounded, with no sign of a disk?
  * 1.1 Smooth? -> GOTO #7
  * 1.2 Featured or disc? -> GOTO #2
  * 1.3 Star or artifact? -> END
2. Could this be a disk viewed edge-on?
  * 2.1 Yes -> GOTO #9
  * 2.2 No -> GOTO #3
3. Is there a sign of a bar feature through the centre of the galaxy?
  * 3.1 Yes -> #4
  * 3.2 No -> #4
4. Is there any sign of a spiral pattern?
  * 4.1 Yes -> #6
  * 4.2 No -> #6
5. How prominent is the central bulge, compared with the rest of the galaxy?
  * 5.1 No bulge -> #6
  * 5.2 Just noticeable -> GOTO #6
  * 5.3 Obvious -> #6
  * 5.4 Dominant -> #6
6. Is there anything odd?
  * 6.1 Yes -> #8
  * 6.2 No -> END
7. How rounded is it?
  * 7.1 Completely round -> #6
  * 7.2 In between -> #6
  * 7.3 Cigar shaped -> #6
8. Is the odd feature a ring, or is the galaxy distrubed or irregular?
  * 8.1 Ring -> END
  * 8.2 Lens or arc -> END
  * 8.3 Disturbed -> END
  * 8.4 Irregular -> END
  * 8.5 Other -> END
  * 8.6 Merger -> END
  * 8.7 Dust lane -> END
9. Does the galaxy have a bulge at its centre? If so, what shape?
  * 9.1 Rounded -> #6
  * 9.2 Boxy -> #6
  * 9.3 No bulge -> #6
10. How tightly wound do the spiral arms appear?
  * 10.1 Tight -> #11
  * 10.2 Medium -> #11
  * 10.3 Loose -> #11
11. How many spiral arms are there?
  * 11.1 1 -> #5
  * 11.2 2 -> #5
  * 11.3 3 -> #5
  * 11.4 4 -> #5
  * 11.5 More than four -> #5
  * 11.6 Can't tell -> #5

Notice that for most of the questions, all responses lead to the same follow-up question. Unlike a typical decision tree, the destination isn't the defining characteristic of the object being analyzed. Rather, it's a process to build a list of attributes that describe a given galaxy.


```python
title = 'Mean Confidence Level of All Classes'
plt.figure(title, figsize=(20, 5))
plt.suptitle(title, fontsize=20)
sns.barplot(x= labels.drop('GalaxyID', axis=1, inplace=False).mean().index,
            y = labels.drop('GalaxyID', axis=1, inplace=False).mean().values)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16, rotation=35, ha='right')
plt.show()
```


![png](/images/galaxy-zoo/output_13_0.png)



```python
title = 'Instances Where Confidence Level is Above 0.5'
plt.figure(title, figsize=(20, 5))
plt.suptitle(title, fontsize=20)
sns.barplot(x= labels.columns[1:] ,
            y = np.where(labels.drop('GalaxyID', axis=1, inplace=False).to_numpy() >= 0.5, 1, 0).sum(axis=0))
plt.yticks(fontsize=16)
plt.xticks(fontsize=16, rotation=35, ha='right')
plt.show()
```


![png](/images/galaxy-zoo/output_14_0.png)



```python
terminals = [2,14,18,19,20,21,22,23,24]
print("Sum of instances of terminal classes where the confidence level is above 0.5.\n")
for n,v in zip([labels.columns[i+1] for i in terminals], [np.where(labels.drop('GalaxyID', axis=1, inplace=False).to_numpy() >= 0.5, 1, 0).sum(axis=0)[i] for i in terminals] ):
  print("%s: %d" % (n,v))
```

    Sum of instances of terminal classes where the confidence level is above 0.5.

    Class1.3: 44
    Class6.2: 53115
    Class8.1: 886
    Class8.2: 4
    Class8.3: 16
    Class8.4: 301
    Class8.5: 202
    Class8.6: 1022
    Class8.7: 27


### Odd or not?


```python
odd = [np.where(labels.drop('GalaxyID', axis=1, inplace=False).to_numpy() >= 0.5, 1, 0).sum(axis=0)[i] for i in [13,14]]
print("Odd: {:,} or {:0.1f}%".format(odd[0], 100*odd[0]/sum(odd)))
print("Not Odd: {:,} or {:0.1f}%".format(odd[1], 100*odd[1]/sum(odd)))

```

    Odd: 8,484 or 13.8%
    Not Odd: 53,115 or 86.2%


Looking at the data, we can see a few things. First, stars and artifacts are extremely rare in this dataset with only 44 intances. Pretty much 0%. This means that we won't be able to build a model that filters out anomalous data. Second, we see that the majority of galaxies are not odd. Only about 14% are odd and go to question 8 to check for type of oddity, and of the odd attributes most are rings, mergers, dust rings, or "other."

There's one question I'd really like to answer. Do the majority of crowdsourced labelers follow the same decision tree? If they do, the correlation matrix should tightly follow the logical progression of the decision tree. Class1.3 will correlate with nothing, Class6.1 (odd feature present) will correlate heavily with Class8.x (types of odd features) and very little with Class6.2 (no odd features), etc. This means looking at 37² feature pairs, which is a handful, but a correlation heatmap can make it more manageable to browse at-a-glance.  


```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

def heatmap(df, title):
    plt.figure(title, figsize=[20,20])
    plt.title(title, fontsize=30)
    df_corr = df.corr()
    #df_corr = np.triu(df_corr, k=1)
    sns.heatmap(df_corr, vmax=0.6, square=True, annot=True, cmap='YlOrRd', cbar=False)
    plt.yticks(rotation = 45)
    plt.xticks(rotation = 45)
    plt.show()

heatmap(labels.corr(), 'Correlation Between Question Answers')

```


    <IPython.core.display.Javascript object>



![png](/images/galaxy-zoo/output_20_1.png)


This heatmap gives an overview of the correlations between all answers to every question.

Here we can see that the relatively rare Class1.3 (which represents stars or other non-galaxies) is correlated with the path through Class1.1 to ambiguous shapes and odd attribtes. Labelers may have confused galaxies with odd appearances and stars. Things like this are a reminder that hand-labeled data doesn't represent the absolute truth, but rather an approximation of the truth. Remember that the challenge associated with this dataset is about modelling the human thought process behind answering these questions.

We can see from the negative correlation between Class6.1 and Class6.2 that disagreement over whether an images had an odd feature was rare. Though we do see some correlation between Class8.3/4/6 features, which means that there was some confusion between disturbed, irregular, and merged galaxies. There were only 16 insances with confidence in Class8.3, 301 with Class8.4, and 1022 with Class8.6. With this in mind, I would prioritize building a model that is effective at correctly predicting Class6.1 and Class8.4 and not worry too much about other the odd features. Class8.x represents a sub-problem that this dataset is not well-equipped to solve, and the loss function should reflect this.

Zooming into individual questions can show us how well humans can discern between different attributes.


```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

groups = [
          ['Class1.1', 'Class1.2', 'Class1.3'],
          ['Class2.1', 'Class2.2'],
          ['Class3.1', 'Class3.2'],
          ['Class4.1', 'Class4.2'],
          ['Class5.1', 'Class5.2', 'Class5.3', 'Class5.4'],
          ['Class6.1', 'Class6.2'],
          ['Class7.1', 'Class7.2', 'Class7.3'],
          ['Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7'],
          ['Class9.1','Class9.2','Class9.3'],
          ['Class10.1', 'Class10.2', 'Class10.3'],
          ['Class11.1','Class11.2','Class11.3','Class11.4','Class11.5','Class11.6']
]

size = 15
fig = plt.figure('Individual Question Heatmaps', figsize=[size,size*3/4])
for i, group in enumerate(groups):
    fig.add_subplot(3, 4, i+1)
    plt.title('Class' + str(i+1) + '.n')
    sns.heatmap(labels[group].corr(), square=True, annot=True,  cmap='YlOrRd', cbar=False)
    plt.xticks([x + 0.5 for x in range(len(group))], labels=[s.replace('Class','') for s in group])
    plt.yticks([y + 0.5 for y in range(len(group))], labels=[s.replace('Class','') for s in group])
    #plt.xlabel(str(i))

fig.tight_layout(pad=3.0)

plt.show()


```


    <IPython.core.display.Javascript object>



![png](/images/galaxy-zoo/output_22_1.png)


Remembering that the answers within each question are mutually exclusive, these individual correlation matrices can show us how much confusion there was over any given feature of a galaxy.

Class6.n shows the ideal case where it is very rare for people to disagree over the galaxies with odd features, while Class9.n reveals that there is fequently confusion over the shape of bulges and even over whether or not a bulge is present.

The less confusion that the human labelers experienced the better accuracy I expect out of the models I'm going to build.


```python
# Examine rows where a given class is above a certain threshold.
show = 5
c = 'Class' + '6.1'
t = 0.5
labels.loc[labels[c] > t].head(show)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GalaxyID</th>
      <th>Class1.1</th>
      <th>Class1.2</th>
      <th>Class1.3</th>
      <th>Class2.1</th>
      <th>Class2.2</th>
      <th>Class3.1</th>
      <th>Class3.2</th>
      <th>Class4.1</th>
      <th>Class4.2</th>
      <th>Class5.1</th>
      <th>Class5.2</th>
      <th>Class5.3</th>
      <th>Class5.4</th>
      <th>Class6.1</th>
      <th>Class6.2</th>
      <th>Class7.1</th>
      <th>Class7.2</th>
      <th>Class7.3</th>
      <th>Class8.1</th>
      <th>Class8.2</th>
      <th>Class8.3</th>
      <th>Class8.4</th>
      <th>Class8.5</th>
      <th>Class8.6</th>
      <th>Class8.7</th>
      <th>Class9.1</th>
      <th>Class9.2</th>
      <th>Class9.3</th>
      <th>Class10.1</th>
      <th>Class10.2</th>
      <th>Class10.3</th>
      <th>Class11.1</th>
      <th>Class11.2</th>
      <th>Class11.3</th>
      <th>Class11.4</th>
      <th>Class11.5</th>
      <th>Class11.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>100123.jpg</td>
      <td>0.462492</td>
      <td>0.456033</td>
      <td>0.081475</td>
      <td>0.000000</td>
      <td>0.456033</td>
      <td>0.000000</td>
      <td>0.456033</td>
      <td>0.000000</td>
      <td>0.456033</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.456033</td>
      <td>0.0</td>
      <td>0.687647</td>
      <td>0.312353</td>
      <td>0.388158</td>
      <td>0.074334</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.213858</td>
      <td>0.473789</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100134.jpg</td>
      <td>0.021834</td>
      <td>0.976952</td>
      <td>0.001214</td>
      <td>0.021751</td>
      <td>0.955201</td>
      <td>0.313077</td>
      <td>0.642124</td>
      <td>0.546491</td>
      <td>0.408711</td>
      <td>0.160096</td>
      <td>0.760688</td>
      <td>0.034417</td>
      <td>0.0</td>
      <td>0.611499</td>
      <td>0.388501</td>
      <td>0.010917</td>
      <td>0.010917</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032377</td>
      <td>0.064143</td>
      <td>0.450225</td>
      <td>0.000000</td>
      <td>0.032377</td>
      <td>0.032377</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021751</td>
      <td>0.207253</td>
      <td>0.152044</td>
      <td>0.187194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.054649</td>
      <td>0.081974</td>
      <td>0.081974</td>
      <td>0.327894</td>
    </tr>
    <tr>
      <th>16</th>
      <td>100263.jpg</td>
      <td>0.179654</td>
      <td>0.818530</td>
      <td>0.001816</td>
      <td>0.573791</td>
      <td>0.244739</td>
      <td>0.047326</td>
      <td>0.197413</td>
      <td>0.016623</td>
      <td>0.228116</td>
      <td>0.071098</td>
      <td>0.067407</td>
      <td>0.106234</td>
      <td>0.0</td>
      <td>0.913055</td>
      <td>0.086945</td>
      <td>0.000000</td>
      <td>0.075167</td>
      <td>0.104487</td>
      <td>0.000000</td>
      <td>0.019174</td>
      <td>0.019174</td>
      <td>0.058436</td>
      <td>0.058436</td>
      <td>0.757836</td>
      <td>0.000000</td>
      <td>0.340376</td>
      <td>0.091809</td>
      <td>0.141605</td>
      <td>0.000000</td>
      <td>0.007855</td>
      <td>0.008768</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016623</td>
    </tr>
    <tr>
      <th>30</th>
      <td>100458.jpg</td>
      <td>0.820908</td>
      <td>0.081499</td>
      <td>0.097593</td>
      <td>0.000000</td>
      <td>0.081499</td>
      <td>0.000000</td>
      <td>0.081499</td>
      <td>0.000000</td>
      <td>0.081499</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.081499</td>
      <td>0.0</td>
      <td>0.921161</td>
      <td>0.078839</td>
      <td>0.355112</td>
      <td>0.465796</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026714</td>
      <td>0.081062</td>
      <td>0.000000</td>
      <td>0.244108</td>
      <td>0.569277</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>100513.jpg</td>
      <td>0.275971</td>
      <td>0.700977</td>
      <td>0.023052</td>
      <td>0.583914</td>
      <td>0.117063</td>
      <td>0.000000</td>
      <td>0.117063</td>
      <td>0.000000</td>
      <td>0.117063</td>
      <td>0.043734</td>
      <td>0.073329</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.881950</td>
      <td>0.118050</td>
      <td>0.008331</td>
      <td>0.108519</td>
      <td>0.159121</td>
      <td>0.025577</td>
      <td>0.000000</td>
      <td>0.025577</td>
      <td>0.000000</td>
      <td>0.104070</td>
      <td>0.726727</td>
      <td>0.000000</td>
      <td>0.525522</td>
      <td>0.058391</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>






```python

```




```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))
print("The highest confidence example for each answer / Feature.")
size = 15
fig = plt.figure('Image Examples', figsize=[size,size*10/4])
#plt.suptitle("Top Image for Each Feature", fontsize=20)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for i, feature in enumerate(labels.columns[1:]):
    fig.add_subplot(10, 4, i+1)
    plt.title(feature + "\n" + desc[i])
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    file = labels.iloc[labels[feature].idxmax()].GalaxyID
    img = plt.imread(DATA + file, format='jpg')
    plt.imshow(img, aspect='auto')
    #plt.xlabel(str(i))
fig.tight_layout(pad=1.0)
plt.show()
```


    <IPython.core.display.Javascript object>


    The highest confidence example for each answer / Feature.



![png](/images/galaxy-zoo/output_28_2.png)


I'm curious to see where the data is located. In the 37 images I pulled it seems like galaxies are generally centered, and usually in the middle but with occasional features that extend to the ege of the images.

I'm going to average 5,000 images to see what it looks like.


```python
arr = np.array([ plt.imread(DATA + pic, format='jpg') for pic in labels.GalaxyID[0:5000]])
plt.imshow(np.average(arr, axis=0).astype(int), aspect='auto')
```




    <matplotlib.image.AxesImage at 0x7f3805dc2358>




![png](/images/galaxy-zoo/output_30_1.png)


With the galaxies centered but features extending to the edge and beyond I don't think I should crop, but I can rotate the images easily without having to center.

# Benchmarking / Simple Models

### Split the data

Separate data for training, testing, and validation. Validation is different than testing because this is a chink of the data that will never be touched in the training and model evaluation steps, as using test data to tune hyperparameters can introduce bias towards those samples.

I've set a global random state constant because it's important to use a consistent random state to ensure the results are the same every time I run this notebook.


```python
X, X_val, y, y_val = train_test_split(labels.GalaxyID,
                                labels[labels.columns[1:]],  
                                test_size = 0.20,
                                random_state = RANDOM  
                                )
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,  
                                                    test_size = 0.20,
                                                    random_state = RANDOM
                                                    )

print("Training set: %d" % X_train.shape[0])
print("Testing set: %d" % X_test.shape[0])
print("Validation set: %d" % X_val.shape[0])
```

    Training set: 39409
    Testing set: 9853
    Validation set: 12316


## Evaluation

Root mean squared error (RMSE) is a popular evaluation metric for regression problems. Alone it is a meaningless number, as it's hard to really quantify a regression problem in the same way as a classification problem. But it's an excellent way to compare the relative performance of different models. By creating a simple benchmark model, we set a frame of reference for the minimum performance.

The competition winner scored 0.07491 root mean squared error (RMSE) and the 50th scored 0.10146. The caveat is that they had more time and resources to tackle this problem... and I have about a week. There is a central pixel benchmark that uses the color of the middle pixel of the image that came in at 0.16194. (My own linear regression central pixel benchmark 0.1564.) If I wanted to place in the top 50% of people who beat that score I'd need to beat 0.12504.

Keep in mind that the human task that we're replicating is a classification problem, but our target is to predict the confidence level that the crowd has for any given answer. That transforms it into a regression problem. Classification isn't viable for this problem because of the difficulty in establishing a ground truth for the features, the Galaxy Zoo is more about bootstrapping data from the crowdsourced estimates.

(Evaluation function moved up to utility functions cell.)

### Benchmark

Extracting the center pixel of each image and saving then to numpy arrays.


```python
ig = image_generator(X_train, batch_size=3000)
init = False

for step in ig:
    if not init:
        centers_train = center_pixel(step)
        init = True;
    else:
        centers_train = np.append(centers_train, center_pixel(step), axis=0)
save(centers_train, 'centers_train.p')
```


```python
ig = image_generator(X_test, batch_size=3000)
init = False
for step in ig:
    if not init:
        centers_test = center_pixel(step)
        init = True;
    else:
        centers_test = np.append(centers_test, center_pixel(step), axis=0)

save(centers_test, 'centers_test.p')
```


```python
print(centers_train.shape)
print(centers_test.shape)
```

    (39409, 3)
    (9853, 3)


Finding the average pixel of each image and saving them to a numpy array.


```python
ig = image_generator(X_train, batch_size=3000)
init = False
averages_train = 0
for step in ig:
    if not init:
        averages_train = average_color(step)
        init = True;
    else:
        averages_train = np.append(averages_train, average_color(step), axis=0)

save(averages_train, 'averages_train.p')
```


```python
ig = image_generator(X_test, batch_size=3000)
init = False
averages_test = 0
for step in ig:
    if not init:
        averages_test = average_color(step)
        init = True;
    else:
        averages_test = np.append(averages_test, average_color(step), axis=0)

save(averages_test, 'averages_test.p')
```


```python
print(averages_train.shape)
print(averages_test.shape)
```

    (39409, 3)
    (9853, 3)


Extracting these values is a time consuming process, so I've saved them to my Drive account for easy retrieval. Everything stored in my Drive is in the directory stored in the SAVE global variable. If you want to run this notebook yourself, you can simply log into your own drive account and swap out the directory variables.


```python
centers_train = load('centers_train.p')
centers_test = load('centers_test.p')
averages_train = load('averages_train.p')
averages_test = load('averages_test.p')
```

### Linear Regression

#### Centers


```python
lr = LinearRegression()
lr.fit(centers_train, y_train)
pred = lr.predict(centers_test)
print("The RMSE is %.4f" % RMSE(pred, y_test))
```

    The RMSE is 0.1564


This outperforms the simple central pixel benchmark by about 3.4%.

#### Averages


```python
lr = LinearRegression()
lr.fit(averages_train, y_train)
pred = lr.predict(averages_test)
print("The RMSE is %.4f" % RMSE(pred, y_test))
```

    The RMSE is 0.1599


This also outperforms the central pixel benchmark, but by a slimmer margin.

#### Sanity check

But what does this mean? I think we need to sanity check this.

I'm going to see what error we get just by computing the RMSE where the prediction for every row in our dataset just the average value for every category.


```python
print("The RMSE is %.4f" % RMSE(
                                np.broadcast_to(
                                    np.array(labels[labels.columns[1:]].mean()),
                                    (labels.shape[0], labels.shape[1]-1)
                                ), np.array(labels[labels.columns[1:]]))
)
```

    The RMSE is 0.1639


Well... The central pixel benchmark only performed 4.6% better than just guessing the average.

So... what I said before about the meaningless of regression metrics on their own and the importance of good benchmarks still holds. It's a complicated issue.

Is the central pixel a bad benchmark? Maybe, or maybe not. But in this situation, we're just grounding ourselves with the worst performance we should expect. Any improvement on that is a gain, and that's what we're trying to maximize here.

*sigh*

This means that we actually have to do some deep reading. TIt turns out that some peopel duge pretty deep into the color bias in this dataset and published a paper, [Galaxy Zoo: the dependence of morphology and colour on environment](https://arxiv.org/abs/0805.2612). They found that color can vary due to density, with red being more dense. In theory, if a galaxy had red spirals it could mean that those spirals are actually spirals and not just a cloud that looks like spirals. But there is a complicating factor: redshift. The researchers found that "Only a small fraction of the colour–density relation is a consequence of the morphology–density relation."

I was curious how color would effect predictions. Since this is an astronomy dataset and the objects we're looking at are travelling at extremely high velocities relative to earth, that there is bias due to redshift. The universe is expanding, and the literal space between things is growing at an aburdly high speed as if it's being created out of nowhere. As galaxies move away from us at relativistic speeds, the light waves stretch out. Low-wavelength, high-frequency light is blue (why the laster that reads a high-density Bluray disc is blue), and large wavelenght light is red. This means that galaxies look redder than they really are, and that older galaxies that have had more time for space to expand are traveling faster and appear even more red.

When looking at only the color data and nothing else to give us context, the information contained in the colors can be biased by a galaxy's age. So the "dense" galaxy that appears red could actually be a more sparse blue galaxy that is really old and far away.

I'm going to make a judgment call here and choose to continue the project using only percieved luminance. Basically black and white, but since we're modelling human perception I'll bias the colors towards human perception of luminance.


# Building the Model

Now it's time to begin the process of building a model.

## Strategy

This is an image classification problem and is well suited to a Covolutional Neural Network (CNN).

Due to the color issues I described here, as well as time and resource constraints, I am going to reduce the dimensionality of the images by collapsing the three color channels into one channel of percieved luminance. This will reduce the input size by 67% right off the bat and dramatically reduce the complexity of the neural network.

I am also going to attempt selecting the region of interest from the images, cropping them, and re-saving at a lower resolution of 128x128. The size of the galazy in proportion to the image is not super relevant because the galaxies are different distances away/ The black space is wasting parameters in the neural network and eating RAM, so it is beneficial to optimize this.

Ideally, we can find achieve good performance for the model while reducing training time.


## Preprocessing

Tight cropping of regions of interest and conversion to grayscale. This was my first pre-processing attempt.


```python
%%capture

for image in labels.GalaxyID:

    im = cv2.imread(DATA + image)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # I read the docs and found this uses human perception of luminance already.
    ret, thresh = cv2.threshold(im, 25, 255, 0)
    # The numbers are upper and lower thresholds
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ROI = (0,0,0,0) # Region of interest
    ROI_area = 0
    for contour in contours: # cv.RETR_LIST exports contrours as a list.
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area > ROI_area:
            ROI_area = area
            ROI = (x,y,width,height)

    x, y, width, height = ROI

    if width > height:
        crop = im[y:y+width,x:x+width]
    else:
        crop = im[y:y+height,x:x+height]

    image = image.replace('jpg','png') # I don't want to multiple the compression loss. meme.jpg.jpg.jpg

    # 64x64
    cv2.imwrite(
        DIR_64 + image, # OpenCV adheres to file extension formats
        cv2.resize(crop, (64,64), interpolation=cv2.INTER_AREA)
    )

    # 128x128
    cv2.imwrite(
        DIR_128 + image,
        cv2.resize(crop, (128,128), interpolation=cv2.INTER_AREA)
    )


!zip -r -j '/content/64.zip' '/content/processed_64x64/'
!cp '64.zip' '/content/gdrive/My Drive/School/CS497/GalaxyZoo/'
!zip -r -j '/content/128.zip' '/content/processed_128x128/'
!cp '128.zip' '/content/gdrive/My Drive/School/CS497/GalaxyZoo/'
```

Re-visited preprocessing and did a wider cropping and exported with color. After poor model performance below the central pixel benchmark, I dediced to revisit the pre-processing and go back to color while also incorporating a less aggressive crop to allow for 360 degree rotation.

Here I'm padding the region of interest 20% on each side as well as increasing the sensitivity for detecting the region of interest.


```python
%%capture

padding_size = 0.2

for image in labels.GalaxyID:

    im = cv2.imread(DATA + image)
    im2 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # I read the docs and found this uses human perception of luminance already.
    ret, thresh = cv2.threshold(im2, 10, 255, 0)
    # The numbers are upper and lower thresholds
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ROI = (0,0,0,0) # Region of interest
    ROI_area = 0
    for contour in contours: # cv.RETR_LIST exports contrours as a list.
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area > ROI_area:
            ROI_area = area
            ROI = (x,y,width,height)

    x, y, width, height = ROI

    if width > height:
        pad = int(width * padding_size)
    else:
        pad = int(height * padding_size)

    if (y-pad >= 0 and
        x-pad >= 0 and
        y + max(width, height) + pad < im.shape[1] and
        x + max(width, height) + pad < im.shape[0]):

        crop = im[y-pad:y+max(width,height)+pad,x-pad:x+max(width,height)+pad]
    else:
        crop = im

    image = image.replace('jpg','png') # I don't want to multiple the compression loss. meme.jpg.jpg.jpg


    # 64x64
    cv2.imwrite(

        DIR_64_WIDE + image, # OpenCV adheres to file extension formats
        cv2.resize(crop, (64,64), interpolation=cv2.INTER_AREA)
    )

    # 128x128
    cv2.imwrite(
        DIR_128_WIDE + image,
        cv2.resize(crop, (128,128), interpolation=cv2.INTER_AREA)
    )

!zip -r -j '/content/64_WIDE.zip' '/content/processed_64x64_WIDE/'
!cp '64_WIDE.zip' '/content/gdrive/My Drive/School/CS497/GalaxyZoo/'
!zip -r -j '/content/128_WIDE.zip' '/content/processed_128x128_WIDE/'
!cp '128_WIDE.zip' '/content/gdrive/My Drive/School/CS497/GalaxyZoo/'
```

#### Explore the processed images


```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))
display(Markdown("### The highest confidence example for each answer / feature - 64x64 grayscale."))
size = 15
fig = plt.figure('Image Examples', figsize=[size,size*10/4])
#plt.suptitle("Top Image for Each Feature", fontsize=20)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for i, feature in enumerate(labels.columns[1:]):
    fig.add_subplot(10, 4, i+1)
    plt.title(feature + "\n" + desc[i])
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    file = labels.iloc[labels[feature].idxmax()].GalaxyID
    img = plt.imread(DIR_64 + file.replace('jpg','png'), format='jpg')
    plt.imshow(img, aspect='auto', cmap='gray')
    #plt.xlabel(str(i))
fig.tight_layout(pad=1.0)
plt.show()
```


    <IPython.core.display.Javascript object>



### The highest confidence example for each answer / feature - 64x64 grayscale.



![png](/images/galaxy-zoo/output_66_2.png)



```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))
display(Markdown("### The highest confidence example for each answer / feature - 64x64 color."))
size = 15
fig = plt.figure('Image Examples', figsize=[size,size*10/4])
#plt.suptitle("Top Image for Each Feature", fontsize=20)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for i, feature in enumerate(labels.columns[1:]):
    fig.add_subplot(10, 4, i+1)
    plt.title(feature + "\n" + desc[i])
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    file = labels.iloc[labels[feature].idxmax()].GalaxyID
    img = plt.imread(DIR_64_WIDE + file.replace('jpg','png'), format='png')
    plt.imshow(img, aspect='auto', cmap='gray')
    #plt.xlabel(str(i))
fig.tight_layout(pad=1.0)
plt.show()
```


    <IPython.core.display.Javascript object>



### The highest confidence example for each answer / feature - 64x64 color.



![png](/images/galaxy-zoo/output_67_2.png)


I'm just starting out learning OpenCV and to be honest I'm pretty surprised that worked on the first try.

Now these may look pixelated, but convolutional neural networks perform well with low resolution images. I'll explain why later.

Normally I would write a preprocessing algorithm that works on the fly, and I am going to do that for rotation, flipping, etc. But for these heavyweight processing steps, real-time preprocessing would create a bottleneck in training and prediction. Instead, my dataframe simply holds the file names and I extracted the region of interest in a giant batch while enjoying a coffee.

## Convolutional Neural Network

#### Setup

Repeat the train_test_split with our new PNG images. The output is guaranteed to be the same because we've got a contant value for random_seed.

It's a good idea repeat data imports at the beginning of a major step so that you can just skip ahead when starting the notebook. You just run the initial setup cells at the top and then jump down to the step you're working on without having to repeat imports. A lot of times I break out steps into their own notebooks, but for this project I'd like to be able to review the whole process in one document. D.R.Y. and modularity are an issue in software development, but not so much in this situation.


```python
labels = load('labels.p')

# I skipped the validation split to get more training data. In the competition,
# the data is validated against a withheld dataset which I do not have.

X_train, X_test, y_train, y_test = train_test_split(labels.GalaxyID.apply(lambda pic: pic.replace('jpg', 'png')),
                                                    labels[labels.columns[1:]].to_numpy(),  
                                                    test_size = 0.20,
                                                    random_state = RANDOM  
                                                    )

print("Training set: %d" % X_train.shape[0])
print("Testing set: %d" % X_test.shape[0])
# print("Validation set: %d" % X_val.shape[0])

def RMSE(pred, truth):
    # Element-wise RMSE score.
    return np.sqrt( np.mean( np.square( np.array(pred).flatten() - np.array(truth).flatten() )))
```

    Training set: 49262
    Testing set: 12316


Load the processed data into local storage from Google Drive


```python
%%capture
#!cp '/content/gdrive/My Drive/School/CS497/GalaxyZoo/64.zip' '/content/64.zip'
#!cp '/content/gdrive/My Drive/School/CS497/GalaxyZoo/128.zip' '/content/128.zip'
!cp '/content/gdrive/My Drive/School/CS497/GalaxyZoo/64_WIDE.zip' '/content/64_WIDE.zip'
#!unzip '/content/64.zip' -d '/content/processed_64x64'
#!unzip '/content/128.zip' -d '/content/processed_128x128'
!unzip '/content/64_WIDE.zip' -d '/content/processed_64x64_WIDE'
```


```python
# Make sure they're all accounted for:
!ls -l /content/processed_64x64_WIDE/ | wc -l
```

    61579


#### Define the Model

In PyTorch, a neural network is defined as a class where each layer is a parameter initialized by a constructor with a forward propagation method.

Input data is propagated forward through the network to make a prediction. Then it is evaluated by an error function, and the error is back-propagated through the network to update the weights.

**Neurons**

The neural network is composed of layers of neurons, which are a simple linear equation wrapped in an activation function. The output of each neuron is connected to one or more neurons in the subsequent layers depending on the architecture of the network.

$z = wx + b$ &emsp; <- The linear equation

Where:
* $z$ = output
* $w$ = weight
* $x$ = input
* $b$ = bias

This information is stored in tensors (where the popular library TensorFlow got its name), which are often implemented as multidimensional arrays. In PyTorch, the tensor inteface is modeled after the Numpy interface. A neuron has individual $w$ and $b$ values for each connection to another neuron in subsequent laters. A neural network is composed of many of these tensors connected in layers or as a graph. In the case of a convolution layer, there is a many to one relationship where outputs of neurons in a local area feed into a single neuron in the next layer. In a densely connected layer, or fully connected layer, each neuron is connected to every neuron in the subsequent layer.

That is the basic idea behind an Artificial Neural Network (ANN) or multilayer perceptron, and other architectures build off of this with different types of laters and structures between layers.

**Activation Functions**

Why can't we just use the linear equation for neurons? The goal of a neural network is to create a non-linear decision boundary, it's basically a non-linear function approximator. But there's an interesting property of linear functions: the composition of two linear functions is always a linear function. If we were to use a linear activation function, no matter how many layers we create in the network the final result will always result in a linear function $w'x + b'$. We end up with a more computationally expensive linear regression model that consolidates a large number of features.

That's neither interesting or useful when we're classifying an image. We need something that explores complex relationships not only between the input features (pixels), but between the layers upon layers of these neurons. Remember that the task we're trying to solve is approximating the process that happens in the human eye and brain between looking at a picture of a galaxy and clicking the answer to a question about that galaxy. To accomplish non-linearity, need to wrap our linear equation in an activation function that breaks linearity.

The most popular activation function for problems like this is a Rectified Linear Unit, or ReLU. This is extraordinarily simple:

$\sigma(x) = max(x,0)$

If the output is greater than zero, we simply use the output of the linear equation. If not, we use zero. Despite being simple, this is the most effective activation function for breaking linearity in a regression problem. It does have a few pitfalls, one being that a neuron can "die" or effectively be zero for any state. This can be solved with a Leaky ReLU, or replacing zero with a tiny multiple of x like this:



$\sigma(x) = max(x,0.005x)$

This isn't a one-size-fits-all solution. For linear regression problems where our target is a probability, like this one, we can use a sigmoid function. This is guaranteed to output a number between 0 and 1.

$\sigma(x)= \frac{1}{1+e^{-z}}$

**Convolution Layers**

Convolution layers composed of kernels, or [image filters](https://en.wikipedia.org/wiki/Kernel_(image_processing)). Filters pass over an image in steps, creating a new image where each pixel is the convolution of the filter matrix and a subsection of the image. Convolution is a matrix operation denoted by an asterisk $*$ that is in essence a weighted combination of two matrices.
It passes in overlapping steps, and the output matrices lose a couple pixels based on the size of the kernel (one pixel on each edge for a 3x3 kernel) unless padding is applied.

These kernels allow the network to analyze local relationships between pixels rather than looking at the entire image.

One typical filter is a sharpening filter, in 3x3 form here, where n is the level of sharpening to be applied.  

|  |  |  |
|---|---|---|
| 0 | -1 | 0 |
| -1 | n | -1 |
| 0 | -1 | 0|

Anyone familiar with image processing will recognize this kernel. But what makes a CNN powerful is that the convolution layers built from neurons will in effect learn new filters that are uniquely fitted to identifying and understanding features in the images. It will likely learn an edge detection filter and other standard filters, but it will also learn unique filters that pick up on subtle things.  

The tensor in a convolution layer can contain multiple kernels. In effect, when we forward propagate through a convolution layer we're creating one new matrix for each kernel.

**Pooling**

When we filter an input matrix through these feature maps, we retain the same dimensionality. But the real power of a convolutional neural network comes from extracting features from an original input image using feature maps, and then extracting subsequently more abstract features from combinations of the outputs.

After each convolution layer, we are increasing the width of our network, that is the number of filtered images. As we add on layers, we begin to suffer from the curse of dimensionality and the number of neurons explodes. It becomes extremely computationally expensive, and carrying that extra data hurts more than helps.

Because our objective with the learned kernels is to recognize patterns, reducing the size of the matrices effectively allows the neural network to zoom out and look at relationships between features recognized at previous convolution steps.

Remember that our ultimate goal is to understand the relationship between the pixels in an image and ultimately turn that information into 37 answers to 11 questions. Imagine looking very close at a picure of a car. Looking through a magnifying glass, you can recognize where edges and smooth surfaces are. Put down the magnifying glass and you can see where those edges make a car company logo, a unique headlight shape, a certain type of wheel. Step away from the photo and you can see the whole car made up of these components. You don't need to see the pixels that define the boundary between the headlight and the hood to understand that the collection of parts you are looking at make either a Toyota or a BMW.

The theory behind a CNN is the same. As information flows through the network, higher level features are being extracted and it's important to recognize a wider variety of features without getting bogged down in tiny details.

Pooling layers reduce the size our our matrices by breaking them down into grids, say 4x4, and reducing them each to a single pixes. There are many strategies, like max, sum, average, etc, and each have their strenghts and benefits.

Max performs kind of like a sharpening filter, and with our grayscale input images I think it will help our network pull out the most important features.

**Fully Connected**

Fully connected layers are flat layers with a full connection to every neuron in the previous and subsequent layers. Where the convolution layers explore local relationships, the fully connected layers begin to look at the whole picture. These layers are basically act as a non-linear function approximator.

The first fully connected layer takes the feautures detected by the convolution layers as input and the final fully connected layer outputs our prediction.

The dense connections in this part of the network are great at analyzing the relationships between features in an image and whatever we are trying to predict, but they do come with a downside. Sometimes they just memorize data and perform amazing on the training data set, but not so well on testing data.

Similar to how the Boston Dynamics robots learn to be more stable on their feet from employees kicking them down, the neural network becomes stronger when we disrupt it by shutting off neurons at random. This causes other neurons which may otherwise not be used for a particular decision to pick up the slack and find alternate ways of making that decision. This is a very common regularization strategy in deep learning.

Another regularization strategy is [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization). This is one of those funny things that works even though people are totally sure *why* it works. This is a PHD level topic, but my basic understanding is that it prevents major shifts in input data from having an outsized effect on the weights and balances. The step of updating the network parameters is called [back propagation](https://en.wikipedia.org/wiki/Backpropagation), where the hill we're descending is a gradient of the loss function with respect to the weights. Batch normalization smooths out this gradient and prevents dramatic changes. The slower and more stable learning process is able to find solutions faster because you don't have to waste steps correcting overshoots and major shifts in the weights.



#### Batch generator

This needed a little modification from the one I used for EDA and benchmarking. PyTorch uses a tensor data structure rather than matrices and


```python
def torch_batches(pics, labels, path=DATA, batch_size=30, rotate=False, fmt='png'):
    '''
    Generate batches of PyTorch tensors from a list of image filenames.
    DATA is a global variable with the path to the image folder.  
    '''
    angles = np.array([0,90,180,270])
    labels = torch.tensor(labels, dtype=torch.float32)
    l = len(pics)
    batches = int(l/batch_size)
    leftover = l % batch_size
    for batch in range(batches):
        start = batch * batch_size
        this_batch = pics[start:start+batch_size]
        batch_labels = labels[start:start+batch_size,:]

        if rotate:
            yield torch.tensor([scipy.ndimage.rotate(
                                plt.imread(path + pic, format=fmt),
                                reshape=False,
                                angle=np.random.randint(0,360)
                                )
                            for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
        else:
            yield torch.tensor([ plt.imread(path + pic, format=fmt) for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
    start = batches * batch_size
    this_batch = pics[start:start+leftover]
    batch_labels = labels[start:start+leftover,:]
    if rotate:
        yield torch.tensor([scipy.ndimage.rotate(
                            plt.imread(path + pic, format=fmt),
                            reshape=False,
                            angle=np.random.randint(0,360)
                            )
                        for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch

    yield torch.tensor([ plt.imread(path + pic, format=fmt) for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
```

PyTorch does not automatically adjust the architecture of neural networks to accomodate the data that you throw at it. The parameters in a Net() class have to have an exact definition of the weights and biases in the network. The following utility function helps calculate the width of convolution and pooling layer outputs. It's only one-dimensional because I'm only using square inputs.


```python
def get_output_width(width, kernel, padding, stride):
    return int((width + 2 * padding - kernel - 1) / stride + 1)
```


```python
'''
Attempt 1:

With 64x64 grayscale input achieved an RMSE of 0.17497 in 180 epochs.
Very bad results, there's a problem with the model.
'''

classes = labels.columns[1:]

in_width = 64

kernel = 3
pool_kernel = 2
padding = int(kernel/2)
stride = 1

c1_in = 1
c1_out = 8
c1_pooled_width = get_output_width(in_width, pool_kernel, padding, pool_kernel)

c2_out = 16
c2_pooled_width = get_output_width(c1_pooled_width, pool_kernel, padding, pool_kernel)

full_1_in = c2_out * c2_pooled_width * c2_pooled_width
full_1_out = int(full_1_in / 8)
full_2_out = int(full_1_out/4)
full_3_out = len(classes)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        These are the layers in the network, and their attributes are the
        weights and biases of the neurons.
        '''
        self.conv1 = nn.Conv2d(in_channels=c1_in,
                               out_channels=c1_out,
                               kernel_size=kernel,
                               stride=stride,
                               padding=padding,
                               padding_mode='zeros') # convolution layer, padding with zeros is convenient because the image background is black  
        self.pool = nn.MaxPool2d(2) # 2x2 kernel, stride of 2 so there's no overlap, and it's a max pooling strategy
        self.conv2 = nn.Conv2d(in_channels=c1_out,
                               out_channels=c2_out,
                               kernel_size=kernel,
                               stride=stride,
                               padding=padding,
                               padding_mode='zeros')
        # we can re-use the pooling step if the strategy stays the same
        self.fc1 = nn.Linear(full_1_in, full_1_out)
        self.fc2 = nn.Linear(full_1_out, full_2_out)
        self.fc3 = nn.Linear(full_2_out, full_3_out)
        # self.dropout = nn.Dropout(p=0.1) # http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf

    def forward(self, x):
        # feed each layer into the next
        x = self.pool(F.relu(self.conv1(x))) # first convolution
        x = self.pool(F.relu(self.conv2(x))) # second convolution
        x = x.view(x.size()[0],-1) # flatten output for fully connected layer
        x = F.relu(self.fc1(x)) # Linear equation wrapped in activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # The output layer
        return x

net = Net()
```


```python
'''
Attempt 2:

Achieved a score of 0.17411. Not good. So bad in fact that I think I've chosen
the wrong loss function.

Attempt 3:

Built a custom loss function using weights based on the distribution of classes.

Result: RMSE of 0.15994. It beats the provided central pixel benchmark, but not my own.
Moving in the right direction, but there are still problems with the architecture.

'''

classes = labels.columns[1:]

in_width = 64

kernel = 5
pool_kernel = 2
padding = int(kernel/2)
stride = 1

c1_in = 3
c1_kernel = 9
c1_out = 64
c1_conv_width = get_output_width(in_width, c1_kernel, int(c1_kernel/2), stride)
c1_pooled_width = get_output_width(c1_conv_width, int(pool_kernel/2), 0, pool_kernel)

c2_kernel = 5
c2_out = 96
c2_conv_width = get_output_width(c1_pooled_width, c2_kernel, int(c2_kernel/2), stride)
c2_pooled_width = get_output_width(c2_conv_width, pool_kernel, int(pool_kernel/2), pool_kernel)

full_1_in = c2_out * c2_pooled_width * c2_pooled_width
full_1_out = int(full_1_in / 8)
full_2_out = int(full_1_out/4)
full_3_out = len(classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        These are the layers in the network, and their attributes are the
        weights and biases of the neurons.
        '''
        self.conv1 = nn.Conv2d(in_channels=c1_in,
                               out_channels=c1_out,
                               kernel_size=c1_kernel,
                               stride=stride,
                               padding=padding,
                               padding_mode='zeros')

        self.pool = nn.MaxPool2d(pool_kernel) # 2x2 kernel, stride of 2 so there's no overlap, and it's a max pooling strategy
        self.conv2 = nn.Conv2d(in_channels=c1_out,
                               out_channels=c2_out,
                               kernel_size=c2_kernel,
                               stride=stride,
                               padding=padding,
                               padding_mode='zeros')
        self.conv2_bn = nn.BatchNorm2d(c2_out)
        # we can re-use the pooling step if the strategy stays the same
        self.fc1 = nn.Linear(full_1_in, full_1_out)
        self.fc2 = nn.Linear(full_1_out, full_2_out)
        self.fc3 = nn.Linear(full_2_out, full_3_out)
        self.dropout = nn.Dropout(p=0.15) # http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf

    def forward(self, x):
        # feed each layer into the next
        x = self.pool(F.relu(self.conv1(x))) # first convolution
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x)))) # second convolution
        x = x.view(x.size()[0],-1) # flatten output for fully connected layer
        x = F.relu(self.dropout(self.fc1(x))) # Linear equation wrapped in activation function
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x) # The output layer
        return x

net = Net()
```


```python
'''
Attempt 4:

I don't know how I missed the signmoid activation function! Probability models
work best when their output is bound by zero and one.  

I also added an additional convolution layer.

Result: RMSE 0.11025

YES! I beat my goal of 0.12504. I'm now 78/326 on the leaderboard.
'''

classes = labels.columns[1:]

in_width = 64


pool_kernel = 2
padding = int(kernel/2)
stride = 1

c1_in = 3
c1_kernel = 9
c1_out = 64
c1_conv_width = get_output_width(in_width, c1_kernel, c1_kernel/2, stride)
c1_pooled_width = get_output_width(c1_conv_width, pool_kernel/2, 0, pool_kernel)
print("C1: ", c1_conv_width)
print('P1: ', c1_pooled_width)

c2_kernel = 5
c2_out = 96
c2_conv_width = get_output_width(c1_pooled_width, c2_kernel, c2_kernel/2, stride)
c2_pooled_width = get_output_width(c2_conv_width, pool_kernel, pool_kernel/2, pool_kernel)
print("C2: ", c2_conv_width)
print('P2: ', c2_pooled_width)

c3_kernel = 3
c3_out = 128
c3_conv_width = get_output_width(c2_pooled_width, c3_kernel, c3_kernel/2, stride)
c3_pooled_width = get_output_width(c3_conv_width, pool_kernel, pool_kernel/2, pool_kernel)
print("C3: ", c3_conv_width)
print('P3: ', c3_pooled_width)

full_1_in = c3_out * c3_pooled_width * c3_pooled_width
full_1_out = 1024
full_2_out = 256
full_3_out = len(classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        These are the layers in the network, and their attributes are the
        weights and biases of the neurons.
        '''
        self.conv1 = nn.Conv2d(in_channels=c1_in,
                               out_channels=c1_out,
                               kernel_size=c1_kernel,
                               stride=stride,
                               padding= int(c1_kernel/2),
                               padding_mode='zeros')

        self.pool = nn.MaxPool2d(pool_kernel) # 2x2 kernel, stride of 2 so there's no overlap, and it's a max pooling strategy
        self.conv2 = nn.Conv2d(in_channels=c1_out,
                               out_channels=c2_out,
                               kernel_size=c2_kernel,
                               stride=stride,
                               padding= int(c2_kernel/2),
                               padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=c2_out,
                               out_channels=c3_out,
                               kernel_size=c3_kernel,
                               stride=stride,
                               padding= int(c3_kernel/2),
                               padding_mode='zeros')
        self.conv2_bn = nn.BatchNorm2d(c2_out)
        self.conv3_bn = nn.BatchNorm2d(c3_out)
        # we can re-use the pooling step if the strategy stays the same
        self.fc1 = nn.Linear(full_1_in, full_1_out)
        self.fc2 = nn.Linear(full_1_out, full_2_out)
        self.fc3 = nn.Linear(full_2_out, full_3_out)
        self.dropout = nn.Dropout(p=0.15) # randomly shut off 15% of neurons in the later  

    def forward(self, x):
        # feed each layer into the next
        x = self.pool(F.relu(self.conv1(x))) # first convolution
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x)))) # second convolution
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(x.size()[0],-1) # flatten output for fully connected layer
        x = F.relu(self.dropout(self.fc1(x))) # Linear equation wrapped in activation function
        x = F.relu(self.dropout(self.fc2(x))) #
        x = F.sigmoid(self.fc3(x)) # The output layer
        return x

net = Net()
```

    C1:  64
    P1:  32
    C2:  32
    P2:  16
    C3:  16
    P3:  8


### Training the Network


```python
loss_weight = torch.tensor(np.sum(y_train, axis=0) / np.sum(y_train))

def weighted_mse_loss(output, targets, weights):
    loss = (output - targets) ** 2
    loss = loss * weights.expand(loss.shape) # broadcast (37,) weight array to (n, 37).
    loss = loss.mean(0)
    return loss.sum()

# optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.7)
optimizer = optim.Adam(net.parameters()) # Adam should converge faster than stochastic gradient descent

loss_history = []
batch_size = 1024
mini_batch_size = 64
display("Start")

for epoch in range(1,6):
    batch_no = 1

    datagen = torch_batches(X_train,
                        y_train,
                        path=DIR_64_WIDE,
                        batch_size=batch_size,
                        rotate=True)

    for images, targets, _ in datagen: # big read from storage
        for mini_batch in range(int(batch_size/mini_batch_size)):
            start = mini_batch*mini_batch_size
            if start >= images.shape[0]:
                break
            finish = start + mini_batch_size
            optimizer.zero_grad()   
            outputs = net(images[start:finish])
            loss = weighted_mse_loss(outputs, targets[start:finish], loss_weight)
            loss.backward()
            optimizer.step()
        clear_output()
        display("Epoch %d, batch %d - loss: %.5f" % (epoch, batch_no, loss.item()))
        batch_no += 1
        loss_history.append((epoch, batch_no, loss.item()))

save(net, 'net_4.p')
torch.save(net, 'net_t_4.p')
save(optimizer, 'optim_4.p')
save(loss_history, 'hist_4.p')
```


    'Epoch 5, batch 50 - loss: 0.01850'


    /usr/local/lib/python3.6/dist-packages/torch/storage.py:34: FutureWarning: pickle support for Storage will be removed in 1.5. Use `torch.save` instead
      warnings.warn("pickle support for Storage will be removed in 1.5. Use `torch.save` instead", FutureWarning)
    /usr/local/lib/python3.6/dist-packages/torch/serialization.py:402: UserWarning: Couldn't retrieve source code for container of type Net. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


#### Evaluating the trained model's performance


```python
datagen = torch_batches(X_test,
                    y_test,
                    path=DIR_64_WIDE,
                    batch_size=1000)

pred = np.empty((0,len(classes)), float)

for images, targets, pics in datagen:
    outputs = net(images)
    pred = np.append(pred, outputs.detach().numpy(), axis=0)

RMSE(y_test, pred)
```

    /usr/local/lib/python3.6/dist-packages/torch/nn/functional.py:1569: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")





    0.11025187953266899



The output of attempt #4 was 0.11025!


```python
import matplotlib.pyplot as plt
title = "Attempt 4 - Loss over batches"
plt.title(title)
plt.xlabel('Batch')
plt.ylabel('Weighted MSE Loss')
plt.plot([error[2] for error in loss_history[3:]])
plt.savefig(FIGS + title +'.png')
plt.show()
```


![png](/images/galaxy-zoo/output_88_0.png)
