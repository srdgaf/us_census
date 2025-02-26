import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

"""
Just adding a quick class that can do some basic operations on the census datasets, allowing for some quick changes to
feature engineering on-the-fly
"""


class CensusPreprocessor:
    def __init__(self, config):
        self.scaler = MinMaxScaler()
        self.oh_encoder = OneHotEncoder()
        self.config = config
        self.feature_names = list(self.config)

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self.feature_names]

        for f, options in self.config.items():
            processing_fn = options["processor"]
            args = [df, f] + (options.get("args") or [])
            kwargs = options.get("kwargs") or dict()
            df = getattr(self, processing_fn)(*args, **kwargs)

        return df

    def ohe(self,
            df: pd.DataFrame,
            col_name: str,
            values_to_drop: list[str] = []) -> pd.DataFrame:
        """Potential issue: if at test/inference, a given category doesn't appear, then this will not create an empty' \
        col of zeros and the pipeline will error. Proper handling to save fitted OHE, but then shouldn't apply per-col.
        TODO: fix.
        """
        df.loc[:, col_name] = df[col_name].astype(str).str.strip()  # For some reason, these lead with a space
        self.oh_encoder.fit(df[[col_name]])
        ohe_values = self.oh_encoder.transform(df[[col_name]])
        categories, names = self.oh_encoder.categories_[0], self.oh_encoder.get_feature_names_out()
        ohe_df = pd.DataFrame.sparse.from_spmatrix(ohe_values).astype(int)
        ohe_df.columns = names

        df = df.drop(columns=[col_name])
        for idx, (cat, nm) in enumerate(zip(categories, names)):
            if cat not in values_to_drop:
                df[nm] = ohe_df.values[:, idx]
                
        return df

    def binary_flag(self,
                    df: pd.DataFrame,
                    col_name: str,
                    values_to_flag: list[str],
                    scale: bool = False,
                    new_feature_name: str = None) -> pd.DataFrame:

        #  Just use the ordinal logic
        return self.ordinal(df, col_name, {1: values_to_flag}, scale=scale, new_feature_name=new_feature_name)

    def ordinal(self,
                df: pd.DataFrame,
                col_name: str,
                values_to_flag_map: dict[int: str],
                unspecified_categoricals_value: int = 0,
                scale: bool = False,
                new_feature_name: str = None) -> pd.DataFrame:
        """
        In the case of wanting a bool/binary flag, just set a single int key in values_to_flag_map. Could have used
        sklean.preprocessing.OrdinalEncoder, but it seems too broad to be as useful as it maybe could be...
        """
        # Inherently dealing with categoricals here, so coerce to string
        df.loc[:, col_name] = df[col_name].astype(str).str.strip()  # For some reason, these lead with a space

        # Check for mapping clash/mapped values are ints
        assert unspecified_categoricals_value not in list(values_to_flag_map)
        assert all([isinstance(x, int) for x in list(values_to_flag_map)])

        # Construct actual pd-compatible map based on concise config map
        unique_values = set(df[col_name].unique())
        pd_mapping = {}
        for k, v in values_to_flag_map.items():
            for cat_value in v:
                pd_mapping[str(cat_value)] = k  # map to str here as well, just to be safe

        missed_items = unique_values - set(pd_mapping.keys())
        for mi in missed_items:
            pd_mapping[mi] = unspecified_categoricals_value

        # Map it!
        df.loc[:, col_name] = df[col_name].map(pd_mapping)

        # Potentially some issues here with train/test, but we're dealing with large datasets, so minimal chance of clipping
        if scale:
            df[[col_name]] = self.scaler.fit_transform(df[[col_name]])

        if new_feature_name:
            df = df.rename(columns={col_name: new_feature_name})

        return df

    def numeric(self,
                df: pd.DataFrame,
                col_name: str,
                scale: bool = False,
                new_feature_name: str = None) -> pd.DataFrame:
        if not scale:
            return df

        df[[col_name]] = self.scaler.fit_transform(df[[col_name]])
        if new_feature_name:
            df = df.rename(columns={col_name: new_feature_name})

        return df

    def save(self, out_fp):
        """
        only arg needed to initialise the class in identical way (until the bundled sklearn modules are fitted in a
        a global fashion -- then those need to be saved as well).
        """
        with open(out_fp, "w") as f:
            json.dump(self.config, f)
