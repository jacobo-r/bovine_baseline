# Work for the 2022 internship at INRIA/DATAIA under the supervision of François CAUD
## This repository contains, in addition to the original code, my work regarding model exploration and testing
### Jupyter Notebooks
#### Most of this code was inspired from multiple guides found here: https://keras.io/examples/vision/
#### Baselines:
* transformerBaseline: CNN feature extractor -> Transformer based architecture classifier (performance not measured in WCE !)
* pure-transformer-baseline: Tubelet Embedding -> positional encoder -> trasnformer based architecture classifier
* 30_models: model used: MobileNetV2 CNN classifier, 30 models created using same model, what changes are the datasets: the nth model uses all the nx10 th frames of all the videos. WCE of the 30 models is plotted by the end of the notebook.
* conv3d-baseline: interesting function available here that allows to rotate an entire video.
* cnn-transformer-baseline: cnn feature extractor -> positional embedding -> transformer classifier
* cnn-logreg: notebook with the final baseline code, emulating RAMP-TEST and plotting the performance of the 11 models.  

#### Tools:
* video_visualization_tool: tool that allows the participants to see multiple frames of the same video at the same time.

#### Exploration:
* dataAugmentation: model used: MobileNetV2 CNN classfier, explored: filtering classes, augmenting number of observations, augmenting the datset via rotations (and other transformations)
* cnn_300thFrame_BinaryFiltered: Same model as for previous Notebook. Binary classification: here the train/test datasets only contain observations from the class A and H.
* baseline_cnn_300thFrame_BinaryGroup: Same model as before. Binary classfiication:No filtering classes, instead we are grouping classes A,B,C,D into one class and E,F,G,H into another one.

### Scripts (available in the submissions folder)
* (In the newBaseLine folder) videoclassifier.py: final product of this internship, submited to RAMP (https://ramp.studio/events/bovine_embryo_survival_prediction_open_2022/leaderboard)

### Report
* My full internship report is available here: see "internship report 2022"
