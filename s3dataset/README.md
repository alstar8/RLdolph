# s3dataset

Адаптация [webdataset](https://github.com/webdataset/webdataset) под наш формат хранения шардов. Наш формат отличается тем, что информация об изображении хранится в .csv файле рядом с архивом, а в оригинальном формате информация хранится в .json или .txt файле внутри архива. Подробнее про их формат и webdataset можно почитать на [гитхабе webdataset](https://github.com/webdataset/webdataset).

### Installation
```bash
git clone https://github.com/ai-forever/s3dataset
cd s3dataset
pip install -r requirements.txt
```

### Basic usage example:
```python
import webdataset as wds
from s3dataset import init_webdataset
from PIL import Image
import numpy as np
import io

storage_options = {
    "anon": False,
    'key': 'your_key',
    'secret':'your_secret_key',
    'client_kwargs': {
        'endpoint_url':'your_endpoint_dataset'
    }
}

init_webdataset(storage_options)


urls = ['s3://s3_path/example.tar']
dataset = wds.WebDataset(
    urls, 
    handler=wds.warn_and_continue
).shuffle(3000)

for c, item in enumerate(dataset):
    # do stuff
    pass
```
С использованием даталоадера:
```python
dataloader = wds.WebLoader(dataset, num_workers=4, batch_size=16)

for c, batch in enumerate(dataloader):
    # do stuff
    pass
```

Поддерживаются практически все методы и функции из оригинальной версии. Больше примеров, а также документацию можно найти в их репозитории.
