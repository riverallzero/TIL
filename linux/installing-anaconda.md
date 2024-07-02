# Installing Anaconda

## 1. download
- check your python version to download the appropriate anaconda(ex.python=3.8)

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
```

## 2. install
```shell
bash Anaconda3-2021.05-Linux-x86_64.sh
```

### result
```shell
Do you accept the license terms? [yes|no]
[no] >>> yes
:
[/home/username/anaconda3] >>> # ENTER
:
You can undo this by running 'conda init --reverse $SHELL'? [yes|no]
[no] >>> yes
```

## 3. check
```shell
conda --version
```

### result
```shell
conda 23.7.4
```

## 4. usage
```shell
conda create -n <env-name>
```

```shell
conda activate <env-name>
```
