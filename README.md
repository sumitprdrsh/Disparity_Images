# Disparity Images
This is a simple computer vision application program that uses a pair of stereo images taken simultaneously from different angles using the stereo camera setup. This concept is widely used to find the depth of different object pixels in an image. The calculation of depth value can then be used in various applications like segmenting object-background pair, applying 3-D effects to an image, applying various filters to an image. This program aims to illustrate the application of disparity and depth images obtained from the input stereo image pair to segment the object and background pixels by applying a blur effect.


## Installation
Pre-Requisite: Python 3.1 or higher
1. Clone the source code into your local directory.
2. Create a virtual environment.
3. Install required libraries (given in requirements.txt) in the virtual environment.

```python
git clone https://github.com/sumitprdrsh/Disparity_Images.git #For cloning the source code in local directory
pip install -r requirements.txt  #Install required libraries
```


## Execution
1. Run the below command in terminal from the project's root directory (Disparity_Images folder).

```python
python src/main.py
```


## Usage
This type of program can be used in mobile phone cameras to find the depth of various objects in an image. The following files can be viewed to observe the image processing pipeline.

```python
> Input images: data/girlL.png and data/girlR.png
> Disparity image: data/disparity_32_51.png
> Depth image: data/depth_32_51.png
> Output image: data/girlL_32_51_output.jpg
> Image Comparison: data/girlL_32_51_table.jpg
```

## Open Issues and Future Scope
1. The code is not refactored yet.
2. The code is not tested on other sets of similar disparity images.
3. The disparity map parameter input is not automated.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
