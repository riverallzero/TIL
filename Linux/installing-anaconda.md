# Installing Anaconda

## 1. download
- check your python version to download the appropriate anaconda(ex.python=3.8)

```
$ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
```

## 2. install
```
$ bash Anaconda3-2023.09-0-Linux-x86_64.sh

Do you accept the license terms? [yes|no]
[no] >>> yes
:
:
You can undo this by running 'conda init --reverse $SHELL'? [yes|no]
[no] >>> yes
```

## 3. check
```
$ conda --version

conda 23.7.4
```

## 4. usage
```
$ conda create -n <env-name>
$ conda activate <env-name>
```
