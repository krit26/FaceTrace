# Standard Imports
import time
import logging

# Third Part Imports

# Internal Imports
from stores.builder import ImageMetadataStoreBuilder, ImageMetadataVectorStoreBuilder


class StoreHolder(object):
    _store_holder, _last_load_time = {}, {}

    @staticmethod
    def get_or_load_store(builder_name, store_name, store_path=None, **kwargs):
        if store_name not in StoreHolder._store_holder:
            if builder_name not in globals():
                raise Exception(f"{builder_name} builder does not exists")

            logging.info("kwargs: {}".format(kwargs))
            builder = globals()[builder_name](store_path, **kwargs)
            StoreHolder._store_holder[store_name] = builder.load(store_path, **kwargs)
            StoreHolder._last_load_time[builder_name] = time.time()

        return StoreHolder._store_holder[store_name]

    @staticmethod
    def get_store(store_name):
        StoreHolder._store_holder.get(store_name, None)
