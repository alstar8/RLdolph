import webdataset
from webdataset.handlers import reraise_exception
from webdataset.tariterators import tar_file_iterator, base_plus_ext, valid_sample
from webdataset import pipelinefilter
import fsspec
import pandas as pd
import io
import re
import random, re, tarfile

trace = False
meta_prefix = "__"
meta_suffix = "__"

def init_webdataset(storage_options, drop_duplicates=True):
    def get_bytes_io(path):
        with fsspec.open(f"simplecache::{path}", s3=storage_options, mode='rb') as f:
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
    
    

def tar_file_iteratorRL(fileobj, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.
    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")
    """
    
    stream = tarfile.open(fileobj=fileobj, mode="r")
    data = stream.extractfile('breakout_0_500.png').read()
    
    for i in stream.getmembers():
        png_name = i.get_info()['name']
        print('PNG_NAME: ',png_name)
        png_name_list = list(png_name)
        position = png_name.find('_')+1
        
        data1 = stream.extractfile(png_name).read()
        
        
        png_name_list[position] = '1'
        png_name2 = "".join(png_name_list)
        print('PNG_NAME2: ',png_name2)
        data2 = stream.extractfile(png_name2).read()
        
        png_name_list[position] = '2'
        png_name3 = "".join(png_name_list)
        print('PNG_NAME3: ',png_name3)
        data3 = stream.extractfile(png_name3).read()
        
        png_name_list[position] = '3'
        png_name4 = "".join(png_name_list)
        print('PNG_NAME4: ',png_name4)
        data4 = stream.extractfile(png_name4).read()
        
        result = dict(fname1=png_name, fname2=png_name2, fname3=png_name3, fname4=png_name4,
                      data1=data1, data2=data2, data3=data3, data4=data4)
        yield result
        stream.members = []
        
        
        if int(png_name.split('_')[1])==1:
            break
            
            
    #del stream
    """
    for tarinfo in stream:
        fname = tarinfo.name
        
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith(meta_prefix)
                and fname.endswith(meta_suffix)):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue  
            
            print('FILE_NAME: ',tarinfo)
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    """            
        
    
def init_webdatasetRL(storage_options, drop_duplicates=True):
    def get_bytes_io(path):
        with fsspec.open(f"simplecache::{path}", s3=storage_options, mode='rb') as f:
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
                    df = df[~df['image_name_0'].duplicated()]
                df.set_index('image_name_0', inplace=True)
                df0 = df.to_dict('index')


                assert isinstance(source, dict)
                assert "stream" in source

                for sample in tar_file_iteratorRL(source["stream"]):
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                    assert (isinstance(sample, dict) and "data1" in sample and "fname1" in sample)
                    
                    sample["__url__"] = url
                    sample.update(df0[sample['fname1']])
                    
                    print('SAMPLE', sample['fname1'])
                    yield sample
            except Exception as exn:
                exn.args = exn.args + (source.get("stream"), source.get("url"))
                if handler(exn):
                    continue
                else:
                    break
            #else:
            #    del df
                
    def group_by_keys(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
        """Return function over iterator that groups key, value pairs into samples.
        :param keys: function that splits the key into key and extension (base_plus_ext)
        :param lcase: convert suffixes to lower case (Default value = True)
        """
        current_sample = None
        for filesample in data:
            assert isinstance(filesample, dict)
            fname1, value1 = filesample["fname1"], filesample["data1"]
            fname2, value2 = filesample["fname2"], filesample["data2"]
            fname3, value3 = filesample["fname3"], filesample["data3"]
            fname4, value4 = filesample["fname4"], filesample["data4"]
            
            
            prefix, suffix = keys(fname1)
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
                current_sample[suffix] = value2
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