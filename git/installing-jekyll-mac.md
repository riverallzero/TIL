# Install Jekyll in Mac OS

1. install home-brew(https://brew.sh)

    ```/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"```

2. run terminal and update installed home-brew
    
    ```brew update```

3. install rbenv: cause mac basically have ruby(ruby 2.6.10p210) but we need higher version(>=3.0)
    
    ```brew install rbenv ruby-build```

4. check version of rbenv can install

    ```rbenv install -l```
    - 3.1.6
    - 3.2.5
    - 3.3.5
    - jruby-9.4.8.0
    - mruby-3.3.0
    - picoruby-3.0.0
    - truffleruby-24.1.0
    - truffleruby+graalvm-24.1.0

5. select version and install

    ```rbenv install 3.2.5```

6. check version of ruby: probably, you can see mac version(ruby 2.6.10p210)

    ```ruby -v```

7. change version of ruby(mac -> rbenv): you have to change system's ruby version

   - ```export PATH="$HOME/.rbenv/bin:$PATH"```
   - ```eval "$(rbenv init -)"```
   - ```ruby -v``` &rarr; ruby 3.2.5(success)

8. move to your directory

    ```cd WORKING-DIR```

9. install bundle

    ```bundle install```

10. execute jekyll server

    ```bundle exec jekyll serve```

11. finish: check http://127.0.0.1:4000/
