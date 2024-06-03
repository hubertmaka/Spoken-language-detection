# Project Documentation: Detecting Spoken Language from an Audio Sample

## 1. Introduction

The "Spoken Language Detection" project aimed to develop an artificial neural network model that could recognize which language is spoken in a given speech sample (e.g. English, German, Polish). Such a system could have a wide range of applications in various fields, such as real-time language recognition in devices using voice assistants for people speaking different languages living in the same household or in telephone exchanges allowing automatic language detection and appropriate routing, speech analysis or speech recognition systems.

## 2. Main Assumptions

The main objectives of the project were:

- Finding and reading the literature on audio processing using artificial neural networks.
- Collecting audio samples in different languages.
- Processing the samples into a suitable form, alignment, augmentation, division into sets.
- Construction and training of a language recognition model.
- Evaluation and optimization of the model.

## 3. Methodology

### 3.1. Initial Familiarization

- **Literature**: In order to achieve the main objectives of the project, the project began by finding and reading articles on convolutional neural networks (CNNs) and digital signal processing (DSP) methods. All articles have been included in the bibliography.

### 3.2. Data Collection

- **Data sources**: Audio data was collected from the free Mozilla Common Voice database created by the Mozilla Foundation.
- **Number of languages**: The data included samples in 3 different languages such as German, French, Polish due to limited hardware resources, however, the system itself was designed to process any number of languages without the need to change the code.

### 3.3. Data Processing

- **Cleaning and validated data**: The data was checked and well-described, so in order to segment the data into sets and use them to teach the model, all we had to do was use the validated.tsv files (where the information on the sound files was located) and clip_duration.tsv (where the duration of the files in milliseconds was located).
- **Segmentation**: From the set of validated files, those whose duration was greater than or equal to the required sample duration were selected. The recordings were then divided into training, validation, and test sets in `60:20:20` ratios, so that a given speaker would not be repeated in different sets, the gender ratio would be `50:50` in each set, and the number of occurrences of a single speaker in a given set would be controlled.
- **Create a data processing pipeline and feature extraction**: In the processing pipeline using `tf.data.Dataset` along with lazy loading, the data were resampled, properly labeled, aligned to a given length, augmented, and a spectrogram was created from them with frequencies on a mel scale and amplitude on a logarithmic scale. In the end, the model was taught on parameters `nfft=2048`, `window=512`, `stdr=256`, and `mels=256` mainly due to hardware limitations, however, with adequate resources one can try to create a more accurate spectrogram. The batch size is `64`. In addition, functions such as `cache()` and `prefetch()` were used for optimization purposes. For a sample length of `6 seconds`, the input spectrogram size is `(64, 375, 256, 1)`.
- **Data augmentation**: A special `ProcessAudio` class has been created, in which functions have been implemented that allow data augmentation. Each training sample was subjected to random Gaussian noise and amplitude variation within a specified range (custom range adjustment is possible). In addition, random samples approximately with probability `0.14` are subjected to one of four augmentations: fading, time masking, pitch shifting and time shifting.

### 3.4. Building the Model

- **Choice of Algorithm**: Used to Convolutional Neural Networks (CNN) architecture, since the language recognition problem was treated as a problem of image classification, which are different spectrograms.
- **Model Architecture**: The final model is as follows:
  
```python
model = keras.Sequential([ 
layers.Input(shape=input_shape), 
layers.Normalization(axis=-1), 
layers.Conv2D(16, (7, 5), activation='relu', padding='same'),
layers.BatchNormalization(), 
layers.MaxPooling2D((3, 2)), 
layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
layers.BatchNormalization(), 
layers.MaxPooling2D((2, 2)), 
layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
layers.BatchNormalization(), 
layers.MaxPooling2D((2, 2)), 
layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
layers.BatchNormalization(), 
layers.MaxPooling2D((2, 2)), 
layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
layers.BatchNormalization(), 
layers.MaxPooling2D((2, 2)), 
layers.Flatten(), 
layers.Dense(256, activation='relu'), 
layers.BatchNormalization(), 
layers.Dropout(0.2), 
layers.Dense(3, activation='softmax') 
])
```

- **Set Size**: The model was trained on a set of `27,000` samples with `9,000` for each language in a `60:20:20` ratio.

### 3.5. Evaluation and Optimization

- **Collection Enlargement**: For better generalization, the training dataset can be increased. This is even advisable as tests of the model on an increasing dataset have shown.
- **More accurate spectrogram**: If one has a good enough GPU and RAM, a more accurate spectrogram can be created.
- **Model augmentation**: With enough data and resources, one can also create more layers with more filters which will certainly improve the model's ability to recognize more complex shapes. You can also try to increase the number of neurons in the Dense layer.

## 4. Results

- **Accuracy of the Model**: The model achieved `68%` accuracy on the test and validation sets.
- **Error Analysis**: The model performed significantly well on languages from different families, e.g. It recognized languages between the Celtic, Germanic or Slavic families better than in the Slavic family.
### **Confusion Matrix**:

&emsp;&emsp; ![confusion_matrix](https://github.com/hubertmaka/Spoken-language-detection/assets/121463460/bd00a007-7bdb-494d-a92c-44ab0a6f504c)

### **Training Plots (Training Ended With Early Stopping)**:

&emsp;&emsp; ![training_plots](https://github.com/hubertmaka/Spoken-language-detection/assets/121463460/7fed7a21-0702-4b0b-9bb7-2e2c68505b79)

### **Comparison different set sizes and validate/train accuracy**:

&emsp;&emsp; ![acc_plots](https://github.com/hubertmaka/Spoken-language-detection/assets/121463460/d25ff2e2-0ba7-4088-8b43-8adb271b6fe7)



## 5. Conclusions

- After testing various configurations and training the model, it can be concluded that the amount of data given to train the model was too small for technical reasons. Increasing it should improve the generalization of the model. You can see it on the 3rd plot.
- Future work can focus on increasing the number of recognized languages and improving accuracy by using more advanced signal processing techniques and deeper neural networks.
- Increasing the accuracy of the spectrogram is also a proposed direction of development.
- A key factor that projected the model testing results obtained was hardware limitations.

## 6. Running the Environment

In order to run my test environment, you must have Docker with the appropriate modules that allow you to use the GPU during learning (if you want to do so). Here is the linkt to the documentation: https://www.tensorflow.org/install/docker. Then follow the following steps:

- Clone the repository:
  ```bash
  git clone https://github.com/hubertmaka/Spoken-language-detection.git
  ```
- Create a folder named “languages” next to the cloned repository:
  ```bash
  mkdir languages
  ``` 
- Put the folders from Mozilla Common Voice into the “languages” folder. The final folder structure should look like this:
  ![image](https://github.com/hubertmaka/Spoken-language-detection/assets/121463460/9493b2a8-00fb-4ebc-8db3-ee9db3d437e3)
- Enter the cloned repository:
  ```bash
  cd ./Spoken-language-detection
  ```
- Create a Docker image:
  ```bash
  git build -t spoken-lang-detection-image [path to Dockerfile (e.g. .)]
  ```
- Run the container:
  ```
  docker run -gpus all -it -p 8888:8888 -v [DIR directory path]:/app -rm spoken-lang-detection-image bash
  ```

## 7. Links

- [TensorFlow Audio Tutorial](https://www.tensorflow.org/io/tutorials/audio)
- [TensorFlow API Docs](https://www.tensorflow.org/api_docs/python/tf)
- FuzzyGCP: A deep learning architecture for automatic spoken language identification from speech signals. Authors: Avishek Garain, Pawan Kumar Singh, Ram Sarkar
- [Introduction to Convolutional Neural Networks](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns?)
- [Spoken Language Recognition on Mozilla Common Voice Part I](https://towardsdatascience.com/spoken-language-recognition-on-mozilla-common-voice-part-i-3f5400bbbcd8)
- [Spoken Language Recognition on Mozilla Common Voice Part II](https://towardsdatascience.com/spoken-language-recognition-on-mozilla-common-voice-part-ii-models-b32780ea1ee4)
- [Spoken Language Recognition on Mozilla Common Voice: Audio Transformations](https://towardsdatascience.com/spoken-language-recognition-on-mozilla-common-voice-audio-transformations-24d5ceaa832b)
- [Spoken Language Identification from Speech Signals](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8478554/)

## 8. Authors

- Hubert Mąka - AGH student of ICT 3rd year



