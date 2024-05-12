# Using Byobu

## about
**byobu** is a terminal multiplexer that makes working in a terminal environment more efficient and convenient.
Terminal multiplexer is a tool that helps you manage multiple virtual terminal sessions simultaneously in a single terminal window, which is useful for handling multiple tasks simultaneously or protecting your work in case of terminal disconnection.

- split sessions: can divide the terminal window into different areas to work on different commands or tasks at the same time.
- background execution: send running tasks to the background and bring them back later to continue working.
- session management: create and manage multiple sessions to run or save different projects or tasks separately.

## 1. donwload(mac)
```shell
$ brew install byobu
```

## 2. usage
1. execution byobu
   - ```$ byobu``` 
   - if you want to split sessions, call byobu seperatly ex) ```$ byobu -S1```, ```$ byobu -S2```

2. running tasks background(add ```&```)
   - ```$ python main.py &```
   - ```$ sh train.sh &```

- check running process
  - ```$ ps -ef```
  - just python: ```$ ps -ef | grep python```

- kill process
   - ```$ kill -9 <process-id>```
   - force: ```$ pkill -f byobu```
