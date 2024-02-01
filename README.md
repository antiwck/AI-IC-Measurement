# AI Model for IC measurement
# The AI model is able to detect the IC and reference object (coin) in video and segment out for post processing to determine the width and legnth of IC in mm.

The program is programmed in Python due to its flexibility and ease of use.
The AI model used is YOLOv8 segmentation model, and the dataset used are annotated in roboflow to segment out the IC and reference coin.
The dataset and model is trained in Google Colab to utilize Google's T4 gpu to train the AI model.

<br />
<p align="center">
  <img src="Sources/Dataset.png" width="800"><br />
  Dataset
  <img src="Sources/Annotation.jpg" width="800"><br />
  Annotation
  <img src="Sources/Segmentation Result.jpg" width="800"><br />
  Segmentation Result
</p>

<br />
1. Result 1
<br />
<p align="center">
  <img src="Sources/Post Processing 1.jpg" width="600"><br />
</p>

<br />
1. Result 1
<br />
<p align="center">
  <img src="Sources/Post Processing 2.jpg" width="600"><br />
</p>

<br />
1. Result 1
<br />
<p align="center">
  <img src="Sources/Post Processing 2.jpg" width="600"><br />
</p>
