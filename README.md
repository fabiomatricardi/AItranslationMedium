# AItranslationMedium
Repocode for Medium Article

This code is tested on Mac Intel
For troubleshooting on installing pytorch refer to the troubleshooting section of my medium article
- https://medium.com/@fabio.matricardi/the-struggles-python-programmers-face-with-development-tools-926df7503f3b
- https://artificialcorner.com/a-7-steps-guide-to-install-and-run-lamini-lm-on-your-computer-7841ea94522b in Section 2

There is a folder called model_it
Change the name at your convenience according to oyur target language
remember to change as well the path in the .py file  in  the `checkpoint`
## Create wirtual environment
```
python3.10 -m venv venv  #version 3.10 is reccomended
source venv/bin/activate #for mac
venv\Scripts\activate  #for windows users
```

## Install packages
With venv activated (see point above)<br>
```
python3.10 -m pip install --upgrade pip
pip  install mkl mkl-include   # required for CPU usage on Mac users  224 Mb
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0  # The core 
```

Other packages required for this project

```
pip install --upgrade pip
pip install transformers
pip install langchain==0.0.173
pip install streamlit
pip install streamlit-extras
```

## In case your model is not a torch weight but Tesnorflow
In case the weights of the model are in .h5 format you need to install tensorflow (like in the example above)
```
pip install tensorflow
```



