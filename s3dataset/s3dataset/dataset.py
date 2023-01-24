import webdataset
from webdataset.handlers import reraise_exception
from webdataset.tariterators import tar_file_iterator, base_plus_ext, valid_sample
from webdataset import pipelinefilter
import fsspec
import pandas as pd
import io
import re


def init_webdataset(storage_options, drop_duplicates=True):
    def get_bytes_io(path):
        with fsspec.open(path, s3=storage_options, mode='rb', skip_instance_cache=True) as f:
            byte_io = io.BytesIO(f.read())
        byte_io.seek(0)
        return byte_io

    def tar2csv(texts):
        for text in texts.split():
            if text[-4:] == '.tar':
                return text[:-3] + 'csv'
    
    s3_regex = re.compile("s3://(.*?)/(.*)$")
    def url_opener(data, handler=reraise_exception, **kw):
        for sample in data:
            url = sample["url"]
            _, bucket, key, _ = s3_regex.split(url)
            try:
                stream = get_bytes_io(url)
                sample.update(stream=stream)
                yield sample
            except Exception as exn:
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                else:
                    break

    def tar_file_expander(data, handler=reraise_exception):
        """Expand a stream of open tar files into a stream of tar file contents.

        This returns an iterator over (filename, file_contents).
        """
        for source in data:
            url = source["url"]
            try:
                if storage_options is None:
                    df = pd.read_csv(tar2csv(url))
                else:
                    df = pd.read_csv(tar2csv(url), storage_options=storage_options)
                
                if drop_duplicates:
                    df = df[~df['image_name'].duplicated()]
                df.set_index('image_name', inplace=True)
                df = df.to_dict('index')

                assert isinstance(source, dict)
                assert "stream" in source

                for sample in tar_file_iterator(source["stream"]):
                    assert (
                        isinstance(sample, dict) and "data" in sample and "fname" in sample
                    )
                    sample["__url__"] = url
                    df_row = df[sample['fname']]
                    sample.update(df_row)
                    yield sample
            except Exception as exn:
                exn.args = exn.args + (source.get("stream"), source.get("url"))
                if handler(exn):
                    continue
                else:
                    break
            else:
                del df
                
    def group_by_keys(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
        """Return function over iterator that groups key, value pairs into samples.
        :param keys: function that splits the key into key and extension (base_plus_ext)
        :param lcase: convert suffixes to lower case (Default value = True)
        """
        current_sample = None
        for filesample in data:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if webdataset.tariterators.trace:
                print(
                    prefix,
                    suffix,
                    current_sample.keys() if isinstance(current_sample, dict) else None,
                )
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
                current_sample.update(filesample)
            if suffix in current_sample:
                raise ValueError(
                    f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
                )
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        if valid_sample(current_sample):
            yield current_sample

    def tarfile_samples(src, handler=reraise_exception):
        streams = url_opener(src, handler=handler)
        files = tar_file_expander(streams, handler=handler)
        samples = group_by_keys(files, handler=handler)
        return samples
    
    webdataset.tariterators.url_opener = url_opener
    webdataset.tariterators.tar_file_expander = tar_file_expander
    webdataset.tariterators.tarfile_to_samples = pipelinefilter(tarfile_samples)
