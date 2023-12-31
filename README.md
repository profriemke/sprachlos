# Sprachlos
![Sprachlos Image](https://github.com/profriemke/sprachlos/blob/main/graphics/sprachlos.png "A sprachlos graphic")
## Description

Sprachlos is a Python-based sign language recognition app. Training data is included and can be expanded to improve accuracy. The data should be placed in the 'data' folder under the respective letters.

## Required Packages
To use Sprachlos, you must first install the necessary packages. You can do this by executing the following command in your command line:
```
pip install pickle cv2 mediapipe numpy tkinter PIL threading random time sklearn os
```

## Steps to Use

**First Step:**
Running `create_dataset.py` will create a new dataset from the training data in the 'data' folder.

**Second Step:**
Running `train_classifier.py` will train the model using the newly created dataset.

**Third Step:**
Now, the model can be used in SPRACHLOS versions 1 and 2. Running `run_Normal.py` will only run the model. SPRACHLOS 1 and 2 integrates the model into an interface and a mini-game.

## License
Anyone can use and modify the model as they wish.

Enjoy!

