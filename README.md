Query Efficiency Improvement in Facial Recognition Database
==
This Repo is for the course project of CS6400 DB Sys Concepts& Design. In this project, we developed a clustering index with K-means for optimizing the query efficiency of large scale image database. 

Acknowledgement
--
We appreciate the help from Arron Guan, with whose work on facial image recognition. We referred some parts of [his work](https://github.com/aaronzguan/Face-Recognition-Flask-GUI) on developing a feature extractor as our baseline in this project. Besides, we also appreciate for the valuable advices from Prof. Sham Navathe and the TAs, Akanksha Kumari and Prashanth Dintyala, during our implementation progress.

Usage
--
### Prerequisite
This project takes advantage of MySQL DBMS for storing and evaluating the performance. Please intall MySQL in advance and modify the user name, password, the name of your created database in ENVS.py according to your personal settings. 

In order to run this repo, please install the dependencies with the file environment.yaml. In our development, we use [Anaconda](https://www.anaconda.com/) to manage our environment. You can install all dependencies with following command:
```bash
conda env create -f environment.yml
``` 
In the demo we uses [CelebA facial image dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for evaluation. Please download the dataset at [this link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and save images under `./data/` forder. 

### Run code 
Please run `main.py` first to extract facial features and initial our benchmark database. Then run `index.py` to start processing server. This stage should be done on a GPU available computer or remote server with `Cuda` support. 

### Demonstration GUI
All files required for the GUI of the demonstration is saved under the folder `./DB_index_demo_client/`. The demo is developed with `node.js`, `react` and a little bit `pug` template. To install all dependencies, we recommend to use [Yarn](https://yarnpkg.com/lang/en/), which is a powerful package management tool.  

Please redirect to the folder `DB_index_demo_client` first, and run command:

```bash
    yarn 
```

There may be a bug from the official dependencies, please replace the file `scrollbarSize.js` under the folder `node-modules/dom-helpers/util/` with the one having the same name in `./DB_index_demo_client/` folder.

Then before you can deploy the demo on a server, you should install webpack and nodemon globally in your computer or server with following command if you haven't:
```bash
    yarn global add webpack
    yarn global add nodemon
```

Finally, to run the demo, please modify all the variable `base_url` to the server url where you deploy your python backend in following files:
```
    ./DB_index_demo_client/src/react/components/DemoPage/DemoPage.jsx
    ./DB_index_demo_client/src/react/components/DemoPage/recognition.jsx
    ./DB_index_demo_client/src/server/api/index.js
```

Finally, build static files with command: 

```bash
    npx webpack
```

And start the server with command:
```
    yarn start
```

If there is any problem you met on deploying this repo, please contact me through my [email address](runze.zhang@gatech.edu).



