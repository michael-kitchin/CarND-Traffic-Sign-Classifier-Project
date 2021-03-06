# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

This project demonstrates traffic sign classification using a convolutional neural network coupled with minimal image pre-processing. Training/validation/test images come from a provided collection of German traffic signs and samples obtained via Google Street View in/around Düsseldorf and Hamburg.

Neural network structural and procedural hyperparameter evaluation was emphasized over image pre-processing techniques.

The steps performed in this project and detailed in the associated report were, as follows:
1. **Load provided images (really: step 0)**
   * Loads training, validation, and test images from pickle files (serialized Python data).
1. **Summarize images**
   * Prints text summaries of dataset and individual image dimensions.
1. **Analyze and display Images**
   * Plots label distribution and samples of each label.
1. **Pre-Process and display Images**
   * Perform simple grayscale and/or normalization pre-processing to (a) constrain training/validation permutations and (b) help me emphasize network structure and hyperparameters.
1. **Build model and architecture**
   * Builds logits and feedback infrastructure. Also includes methods for training and evaluating the model using batched inputs.
1. **Train & evaluate vs validation set**
   * Automates (a) building the model with a spread of key structural/procedural hyperparameters and (b) evaluating it, repetitively to isolate the best example.
1. **Evaluate vs test set**
   * Evaluates the best, trained model vs a provided test set of images.
1. **Load and display new images**
   * Loads and pre-process ad hoc images taken from Google Street View in/around Düsseldorf.
1. **Evaluate vs new images and analyze performance**
   * Evaluates the best, trained model vs above images, the analyze performance by printing/plotting label prediction for each.

These map approximately to the project rubric points, as detailed below.

## Rubric Points

### See: [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)  

---
### 1. Files Submitted

#### 1.1 Submission Files

_The project submission includes all required files._

See: [GitHub Repo](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project)

* The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code cells executed and displaying output.
   * [Traffic_Sign_Classifier.ipynb](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* An HTML or PDF export of the project notebook with the name report.html or report.pdf.
   * [report.html](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html)
* Any additional datasets or images used for the project that are not from the German Traffic Sign Dataset.
   * [input_data/real_signs](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/tree/master/input_data/real_signs) (attribution: Google Street View)
* Your writeup report as a markdown or pdf file.
   * [README.md](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/README.md) (this file)
---
### 2. Dataset Exploration

#### 2.1 Dataset Summary
         
_The submission includes a basic summary of the data set._

Once loaded, basic metrics about the provided images and sets were printed for verification and reference.

```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Array data shape = (34799, 32, 32, 3)
Number of classes = 43
```
      
#### 2.2. Exploratory Visualization
          
_The submission includes an exploratory visualization on the dataset._

A distribution of image vs label counts and samples of each label were plotted for verification and reference.

Plot:

![Plot: Images vs Labels](./media/images-vs-labels-1.png)

Examples:

![Label #1 Example](./media/label-0-example-1.png)
![Label #2 Example](./media/label-1-example-1.png)
![Label #3 Example](./media/label-2-example-1.png)

See [report.html](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) for remainder.

---
### 3. Design and Test a Model Architecture

#### 3.1 Preprocessing

_The submission describes the preprocessing techniques used and why these techniques were chosen._

After this, parameterized pre-processing functions were developed to enable multiple, repetitive evaluations. Options were limited to grayscaling and amplitude normalization.

Samples of every label and pre-processing option were plotted for verification and reference.

Examples:

![Preprocessing Examples #1](./media/preprocessing-examples-1.png)
      
See [report.html](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) for remainder.

#### 3.2 Model Architecture
         
_The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged._

LeNet-5 was chosen as a baseline model architecture, both due to its established reputation in image classification and familiarity from earlier in this course. 

Model construction was encapsulated and parameterized to enable evaluation of both both procedural (e.g., batch size, epoch count, dropout fraction, learning rate, initialization parameters) and structural (e.g., convolution layer depth) hyperparameters. This required tracking these hyperparameters in order to instantiate models compatible with the dimensionality of the best weights discovered/saved during training.

Developing and evolving this modularity imparted valuable understanding of the TensorFlow API and graph lifecycle.

Summary:

| Layer                 | Description
| :--------------------:|----------
| Input                 | Input: 32x32x(n1) RGB image<br/>Where (n1) = configured image depth (1 or 3)
| Convolution #1        | Input: 32x32x(n1), 1x1 stride, valid padding<br/>Output: 28x28x(n2)<br/>Where (n2) = configured convolution depth #1 (best: 24) 
| Relu (Activation) #1  |
| Pooling #1            | Input: 28x28x(n2), 2x2 kernel size/stride, valid padding<br/>Output: 14x14x(n2)
| Convolution #2        | Input: 14x14x(n2), 1x1 stride, valid padding<br/>Output: 10x10x(n3)<br/>Where (n3) = configured convolution depth #2 (best: 64) 
| Relu (Activation) #2  |  
| Pooling #2            | Input: 10x10x(n3), 2x2 kernel size/stride, valid padding<br/>Output: 5x5x(n3)
| Connected #1          | Input: 5x5x(n3) weights<br/>Output: 120 weights
| Relu (Activation) #3  |  
| Connected #2          | Input: 120 weights<br/>Output: 84 weights
| Relu (Activation) #4  |  
| Connected #3          | Input: 84 weights<br/>Output: (n4) weights<br/>Where (n4) = # of classes (default: 43)

#### 3.3. Model Training
          
_The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters._

Training and verification steps were evaluated over multiple runs (~2000) using automatically-generated ranges of hyperparameters. Permutations were shuffled and repeated to minimize unanticipated, cumulative effects.

`AdamOptimizer` was selected due to its general robustness, simplicity, and attractive traits such as (apparently) built-in, interactive learning rate reduction.

This effort yielded valuable experience, but may have been over-emphasized due to general unfamiliarity with the domain and TensorFlow, specifically.   
      
#### 3.4 Solution Approach
         
_The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater._

In addition to the systematic evaluation of hyperparameters, described above the training algorithm optimized execution, as follows:
* Validation accuracy and related configurations were tracked and ranked at conclusion of all runs for verification and reference
* Weights and configurations were stored whenever improvement on previous, best accuracy was identified, regardless of current epoch
* Training runs were terminated whenever accuracy consistently degraded (i.e., less than observed best in a given run for (n) epochs) 

##### 3.4.1 Validation Performance
This approach routinely produced validation accuracy greater than 0.98.

The best hyperparameters discovered through this process were subsequently made training algorithm defaults and are printed during execution and in rankings at conclusion:
```
Total passes 3 (tries=3, permutations=1):
Current pass # 1 of 3
    curr_mu:  0.0
    curr_sigma:  0.1
    curr_rate:  0.0013
    curr_epochs:  100
    curr_keep:  0.5
    curr_batch_size:  128
    curr_grayscale:  True
    curr_normalize:  True
    curr_conv1_depth:  24
    curr_conv2_depth:  64
Stopping at epoch # 43 (degrading accuracy)!
Accuracy (Validation) = 0.975

[...]

Top 3 results:
Accuracy = 0.980 / Config = [0.0, 0.10000000000000001, 0.0012999999999999999, 67, 0.5, 128, True, True, 24, 64]
Accuracy = 0.975 / Config = [0.0, 0.10000000000000001, 0.0012999999999999999, 22, 0.5, 128, True, True, 24, 64]
Accuracy = 0.970 / Config = [0.0, 0.10000000000000001, 0.0012999999999999999, 8, 0.5, 128, True, True, 24, 64]
```

Summary:

| Field | Output Position | Definition | Best Value
|-------|:---------------:|------------|:---------:
| `curr_mu` | 1st | Mean for randomly-initialized weights | 0.0
| `curr_sigma` | 2nd | Standard deviation for randomly-initialized weights | 0.1
| `curr_rate` | 3rd | Initial learning rate (`AdamOptimizer` is said to decay this) | 0.0013
| `curr_epochs` | 4th | Maximum epochs shown at start<br/>Epoch with best accuracy shown in ranking | 100 (or) Best
| `curr_keep` | 5th | Weight retention probability during training (i.e., dropout) | 0.5
| `curr_batch_size` | 6th | Evaluation batch size | 128
| `curr_grayscale` | 7th | Grayscale images before evaluation? | True
| `curr_normalize` | 8th | Normalize images before evaluation? | True
| `curr_conv1_depth` | 9th | Convolution layer #1 depth | 24
| `curr_conv2_depth` | 10th | Convolution layer #2 depth | 64

See [Traffic_Sign_Classifier.ipynb](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) for hyperparameter evaluation ranges (commented-out).

##### 3.4.2 Test Performance 
Test accuracy was routinely greater than 0.95.
```
Accuracy (Test) = 0.950673000641
```

See [report.html](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) for more detailed results.

---
### 4. Test a Model on New Images
       
#### 4.1 Acquiring New Images

_The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify._

Ten new images of signs were captured via screenshot from Google Street View, taken from in/around Düsseldorf and Hamburg. No consideration was made for sign type vs training set sign type (label) distribution, plotted above.

In general, these images appeared clearer (superior color fidelity, brightness, and/or contrast) and/or acquired from less-oblique angles than those in the provided training set. These qualities were anticipated to work against the model's trained characteristics, despite being more usable for human decision-making. 

Image labels were identified via provided `signnames.csv` file and incorporated into the filenames, as shown:
```
$ ls
label-13.png  label-1.png   label-25.png  label-33.png  label-5.png
label-17.png  label-22.png  label-2.png   label-35.png  label-7.png
$
```
Summary:

|Image|Label|Notes|
|-----|:-----:|-----|
|![Image #1](input_data/real_signs/label-17.png)|17|Poor brightness/contrast but w/greater saturation than examples may defeat contrast-/brightness-driven affinities.|
|![Image #2](input_data/real_signs/label-7.png)|7|Good color/brightness/contrast; Cross-cutting glare may defeat brightness-driven affinities.|
|![Image #3](input_data/real_signs/label-13.png)|13|Good color/brightness/contrast; Upper corner clipping may defeat edge-driven affinities.|
|![Image #4](input_data/real_signs/label-35.png)|35|Good color/brightness/contrast; Adjoining, partly-overlapping sign below may defeat edge-driven affinities.|
|![Image #5](input_data/real_signs/label-1.png)|1|Average color/brightness/contrast; Adjoining, partly-overlapping sign above may defeat edge-driven affinities.|
|![Image #6](input_data/real_signs/label-5.png)|5|Degraded lines from over-saturation may defeat edge-driven affinities.|
|![Image #7](input_data/real_signs/label-2.png)|2|Average color/brightness/contrast; Degraded lines from motion/depth-of-field blur may defeat edge-driven affinities.|
|![Image #8](input_data/real_signs/label-22.png)|22|Good color/brightness/contrast; Lower corner clipping may defeat edge-driven affinities.|
|![Image #9](input_data/real_signs/label-33.png)|33|Good color/brightness/contrast; Good candidate.|
|![Image #10](input_data/real_signs/label-25.png)|25|Poor brightness/contrast; Lower corner clipping may defeat edge-driven affinities.|
    
#### 4.2 Performance on New Images
       
_The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set._

At first, performance on new images proved very low (0.10), approximate to random results. The cause proved to be inconsistencies between image color models loaded from provided training/validation/test pickle files vs acquired PNG files. 

This was due to use of `matplotlib` instead of `cv2` for image loading, due (in turn) to unavailability of `cv2` on the provided AMI. Evaluation and resolution of this concern delayed project completion for at least a week. 

See [Issue Report](https://github.com/udacity/sdc-issue-reports/issues/768) for additional information.

Once resolved, performance on new images was routinely greater than 0.90.
```
Accuracy (Real) = 0.900
```
This reduction in accuracy vs validation (0.98)/test (0.90) images, above suggests over-fitting and/or counter-productive bias in training/validation/test sets. It's also reasonable to assume identified label probability distribution for given images (see below) may be correlated with training set label distribution (plotted above).    

#### 4.3 Model Certainty - Softmax Probabilities
         
_The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions._

The top five softmax preditiction probabilities were evaluated, printed, and plotted on logarithmic scales for reference and verification. 

In general, the devloped model was drastically more certain of its best choice than nearest alternatives, with 1st-to-2nd choice multiples routinely greater than 10e2.

Examples:

![Softmax Plot Example #1](./media/softmax-plot-example-1.png)
![Softmax Plot Example #2](./media/softmax-plot-example-2.png)

See [report.html](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) for more detailed results.

---
### Visualizing Working State

Activation layer working state was also printed and plotted for verification and reference using the supplied `outputFeatureMap` function.

This proved usable for only one of the activation layers, however (the first).

Examples:

![Working State Example #1](./media/working-state-example-1.png)

See [report.html](https://github.com/michael-kitchin/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) for more detailed results.
