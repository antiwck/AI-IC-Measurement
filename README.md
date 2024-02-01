# AI Model for IC measurement
# The AI model is able to detect the IC and reference object (coin) in video and segment out for post processing to determine the width and length of IC in mm.

The program is programmed in Python due to its flexibility and ease of use.
The AI model used is YOLOv8 segmentation model, and the dataset used are annotated in roboflow to segment out the IC and reference coin.
The dataset and model is trained in Google Colab to utilize Google's T4 gpu to train the AI model.

<br />
<p align="center">
  <img src="Sources/Dataset.png" height="300"><br />
  Dataset<br />
  <img src="Sources/Annotation.jpg" height="300"><br />
  Annotation<br />
  <img src="Sources/Segmentation Result.jpg" height="300"><br />
  Segmentation Result
</p>

<br />
1. Result 1
<br />
<p align="center">
  <img src="Sources/Post Processing 1.jpg" height="300"><br />
  IC is detected regardless of how damaged it is. <br />
</p>

<br />
2. Result 2
<br />
<p align="center">
  <img src="Sources/Post Processing 2.jpg" height="300"><br />
  IC is detected given that both IC and coin are in perfect condition. <br />
</p>

<br />
3 Result 3
<br />
<p align="center">
  <img src="Sources/Post Processing 3.jpg" height="300"><br />
  IC cannot be measured due to missing reference object. <br />
</p>
