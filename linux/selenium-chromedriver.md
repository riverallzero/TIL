# Selenium & Chromedriver

##  Install Chrome
- ```cd /tmp```
- ```wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb```
- ```sudo apt install ./google-chrome-stable_current_amd64.deb```
- ```sudo mv /usr/bin/google-chrome-stable /usr/bin/google-chrome```
- ```google-chrome --version``` 

if you can check chrome version like this &rarr; ```126.0.6478.126``` , you're good to go

## Install Chromedriver
before install, please move to code's directory &rarr; ```cd <dir>``` and download your version's driver
- ```wget https://storage.googleapis.com/chrome-for-testing-public/126.0.6478.126/linux64/chromedriver-linux64.zip```
- ```unzip chromedriver-linux64.zip```
- ```rm -rf chromedriver-linux64.zip```

## Install Selenium
- ```pip install selenium```

## Test
```python
from selenium import webdriver as wd
import time

driver = wd.Chrome()
driver.get('google.com')
time.sleep(20)
```
