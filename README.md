# Temporally Coherent Full 3D Mesh Human Pose Recovery from Monocular Video
This repository contains the data generation for ["Temporally Coherent Full 3D Mesh Human Pose Recovery from Monocular Video"](https://arxiv.org/abs/1906.00161)

Click image to watch in Youtube:

[<img src="https://github.com/liujianee/MVIPER/blob/master/assets/female_106_12_Full_with_Music.gif" width="40%">](https://youtu.be/Olbo53PgGH8)


## Environment
- Ubuntu 16.04
- Python 3.5
- Blender 2.79
- MarvelousDesigner


## Usage

### ENV SETUP
1. Install [Blendre-2.79](https://www.blender.org/download/).
2. Install [Marvelous Designer](https://www.marvelousdesigner.com/product/pricing/).
3. Download [smpl-data](https://drive.google.com/drive/folders/11tulnh1hdrMNA4ABWqj4Bzli2HVw1kkR?usp=sharing).


### GENERATE ANIMATION

1. check settings in [config](https://github.com/liujianee/MVIPER/blob/master/datageneration/config).
2. run [script](https://github.com/liujianee/MVIPER/blob/master/datageneration/run_Generate_Animation.sh).
3. get animation sequence without cloth in [here](https://github.com/liujianee/MVIPER/tree/master/datageneration/animate_out).

### CLOTH SIMULATION

1. use [MD](https://www.marvelousdesigner.com/) to design [cloth](https://github.com/liujianee/MVIPER/tree/master/datageneration/MD_Assets) for avators.
2. put cloth on and run cloth simulation for a target animation.
3. save cloth mesh and motion files in [here](https://github.com/liujianee/MVIPER/tree/master/datageneration/animate_out).
4. you can download simualted action with cloth files from [here](https://drive.google.com/drive/folders/11tulnh1hdrMNA4ABWqj4Bzli2HVw1kkR?usp=sharing).

### RENDER ANIMATION

1. collect your preferred backgrounds and textures, place them in appropriate [path](https://github.com/liujianee/MVIPER/blob/master/datageneration/config).
2. run [script](https://github.com/liujianee/MVIPER/blob/master/datageneration/run_Render_Animation.sh).
3. you can download some rendered data from [here](https://drive.google.com/drive/folders/11tulnh1hdrMNA4ABWqj4Bzli2HVw1kkR?usp=sharing).

### EXAMPLES

<img src="https://github.com/liujianee/MVIPER/blob/master/assets/female_05_05_Full.gif" width="40%">

<img src="https://github.com/liujianee/MVIPER/blob/master/assets/female_05_06_Full.gif" width="40%">

<img src="https://github.com/liujianee/MVIPER/blob/master/assets/male_05_10_Full.gif" width="40%">

<img src="https://github.com/liujianee/MVIPER/blob/master/assets/male_104_16_Full.gif" width="40%">

<img src="https://github.com/liujianee/MVIPER/blob/master/assets/male_85_01_Full.gif" width="40%">



## References
- [surreal](https://github.com/gulvarol/surreal)

