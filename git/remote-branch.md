# Manage Remote Branch in Local

how to get remote branch in repository?

first, we need to update git branch.

```shell
git remote update
```

this code updates all the remote branch information in local repository.
also can check which branch you have. 

- ```git branch -r``` &rarr; show branch list in remote
- ```git branch -l``` &rarr; show branch list in local
- ```git branch -a``` &rarr; show branch list in remote and local
- ```git branch -v -a``` &rarr; show branch list and history of all branch

if you want to bring remote branch to local, use ```git checkout -t```

let's assume we have ```example``` branch in remote repository
```shell
git branch -a

* main
  remotes/origin/example
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
```

what we need is track remote branch and make new branch in local
```shell
git checkout -t remotes/origin/example
```

then we can get local branch!!
```shell
git branch -a

* example
  main
  remotes/origin/example
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
```

after edit your code, staging your file and use ```git push origin <branch-name>``` for push

```shell
git add .
git commit -m 'commit-message'
git push origin example
```
