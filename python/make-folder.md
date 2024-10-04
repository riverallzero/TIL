# os.makedirs()

When we want to save file in directory, error occured if we don't have.
In this case we can use ```os.makedirs()``` to make the directory.

```python
import os

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

Then, using ```os.path.join(output_dir, filename)``` can save file in directory we made.

```python
 df.to_csv(os.path.join(output_dir, 'output.csv'), index=False)
```
