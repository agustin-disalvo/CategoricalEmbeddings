
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import tqdm
import time


class CatEmbedder():

    def __init__(self,
                 embedding_size=32,
                 embedding_task="clas",
                 random_state=5,
                 verbose=False):

        warnings.simplefilter("ignore")

        self.embedding_size = embedding_size
        self.random_state = random_state
        self.verbose = verbose
        self.embedding_task = embedding_task

        self.num_feats = 0
        self.name_feats = None
        self.embedding_feats = None

        self.embedding_loss = None
        self.ohe_vectors = None
        self.embeddings = None
        self.embedding_pca = None
        self.embedding_umap = None

        self.scaler_embedder = None
        self.pca_embedder = None
        self.umap_embedder = None

    def _check_data(self, feats, target):

        """
        DESCRIPTION: Internal function for checking data consistency.
        PARAMS:
          * feats (series, dataframe): Features to embed
          * target (series, dataframe): Target to embed features against
        RETURN:
          * (dataframes) Features and target processed and validated
        """

        # Check data alignment
        assert feats.shape[0] == target.shape[0], \
            f"Features and target have different number of observations: {feats.shape[0]} != {target.shape[0]}"

        # Check data type
        if isinstance(feats, pd.core.series.Series):
            feats = pd.DataFrame(feats)

        # Check feat data type
        for i in feats.columns:

            if feats[i].dtypes == int:

                # If numeric and high cardinality, raise warning
                if feats[i].nunique() / feats.shape[0] > 0.8:
                    warnings.warn(
                        f"Feature '{i}' is numeric and has high cardinality ¿do you want to include it?")

        return feats, target

    def fit(self, embedding_features, embedding_target):

        """
        DESCRIPTION: Function to generate embeddings
        PARAMS:
          * embedding_features (dataframe): Processed features
          * embedding_target (dataframe): Processed target
        RETURN:
          * Embedding representations: Original, PCA and UMAP
        """
        start = time.time()

        print("* Processing features...") if self.verbose else None

        # Data validation
        features, target = self._check_data(embedding_features,
                                            embedding_target)

        self.embedding_feats = features

        # One hot enconding features
        ohe_feats = [pd.get_dummies(features[i]) for i in features.columns]

        self.name_feats = [i for i in features.columns]
        self.num_feats = len(ohe_feats)

        # If embedding more than one feature
        if self.num_feats > 1:

            # Concatenate OHE features
            ohe_features = pd.concat(ohe_feats, axis=1)

            # Generate vector mapping
            ohe_vectors = ohe_features.drop_duplicates()

            # Naming the vectors requires concatenating the names of the one-hot encoded features
            # on every occurence.
            ohe_vectors.loc[:, "VEC_NAMES"] = ohe_vectors.apply(lambda x: "-".join(
                [i for i in x[x == 1].index]
            ),
                                                                axis=1)
            ohe_vectors = ohe_vectors.set_index("VEC_NAMES")
            ohe_vectors = ohe_vectors.sort_values(by=list(ohe_vectors.columns),
                                                  ascending=False)

        # If embedding only one feature
        else:

            # Table extraction
            ohe_features = ohe_feats[0]

            # Generating vector map
            ohe_vectors = ohe_features.drop_duplicates()
            ohe_vectors = ohe_vectors.sort_values(by=list(ohe_features.columns),
                                                  ascending=False).reset_index(drop=True).T

        self.ohe_vectors = ohe_vectors

        # Define the model according to the embedding task
        if self.embedding_task == "clas":
            emb_model = MLPClassifier(hidden_layer_sizes=(self.embedding_size,),
                                      random_state=self.random_state)

        elif self.embedding_task == "reg":
            emb_model = MLPRegressor(hidden_layer_sizes=(self.embedding_size,),
                                     random_state=self.random_state)

        else:
            raise ValueError(f"'{self.embedding_task}' is not available as an embedding task, try 'reg' or 'clas'")

        print("* Training model...") if self.verbose else None

        # Model fitting and embedding extraction
        emb_model.fit(ohe_features, embedding_target)
        self.embedding_loss = emb_model.loss_

        print(f"* Embedding loss: {self.embedding_loss:.4f}") if self.verbose else None

        self.embeddings = pd.DataFrame(ohe_vectors.values @ emb_model.coefs_[0],
                                       index=ohe_vectors.index)

        print("* Generating embeddings...") if self.verbose else None

        # Processing data for calculating embedding representations
        scaler_ = StandardScaler()

        pca_ = PCA(n_components=0.95,
                   random_state=self.random_state)

        umap_ = UMAP(n_components=2,
                     metric="cosine",
                     random_state=self.random_state,
                     min_dist=0.3,
                     n_neighbors=40)

        emb_sc = scaler_.fit_transform(self.embeddings)
        print("  - Scaling ready") if self.verbose else None

        emb_pca = pca_.fit_transform(emb_sc)
        print("  - PCA ready") if self.verbose else None

        emb_umap = umap_.fit_transform(emb_pca)
        print("  - UMAP ready") if self.verbose else None

        # Assignment
        self.embedding_pca = pd.DataFrame(emb_pca, index=ohe_vectors.index)
        self.embedding_umap = pd.DataFrame(emb_umap, index=ohe_vectors.index)

        self.scaler_embedder = scaler_
        self.pca_embedder = pca_
        self.umap_embedder = umap_

        end = time.time()
        print(f"Total time: {np.round((end - start) / 60, 2)} minutes.")

    def transform(self, use_representation="pca"):

        if self.num_feats > 1:
            ix = pd.Index(
                self.embedding_feats.apply(lambda x: "-".join([str(i) for i in x]),
                                           axis=1).values
            )
        else:
            ix = pd.Index(self.embedding_feats.iloc[:, 0].values)

        if use_representation == "raw":
            res = self.embeddings.reindex(ix)

        elif use_representation == "pca":
            res = self.embedding_pca.reindex(ix)

        elif use_representation == "umap":
            res = self.embedding_umap.reindex(ix)

        else:
            raise ValueError(f"'{use_representation}' is not available as an output representation try 'raw', 'pca' or 'umap'")

        res.columns = [f"EMB_{'-'.join(self.name_feats)}_C{i}" for i in range(res.shape[1])]
        return res

    def plot_embeddings(self):

        """
        DESCRIPTION: Two dimensional embedding plotting
        PARAMS: None
        RETURN:
          * Plot
        """

        fig, ax = plt.subplots(figsize=(12, 9))

        ax.scatter(self.embedding_umap.iloc[:, 0],
                   self.embedding_umap.iloc[:, 1])

        for ix, i in enumerate(self.embeddings.index):
            plt.annotate(i, (self.embedding_umap.iloc[ix, 0],
                             self.embedding_umap.iloc[ix, 1]))

        plt.show()
