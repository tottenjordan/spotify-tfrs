
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

import train_config as cfg
# ====================================================
# Playlist (query) Tower
# ====================================================

# TODO: parameterize

EMBEDDING_DIM = cfg.EMBEDDING_DIM       # 32
PROJECTION_DIM = cfg.PROJECTION_DIM     # 5
SEED = cfg.SEED                         # 1234
USE_CROSS_LAYER = cfg.USE_CROSS_LAYER   # True
DROPOUT = cfg.USE_DROPOUT               # 'False'
DROPOUT_RATE = cfg.DROPOUT_RATE         # '0.33'
MAX_PLAYLIST_LENGTH = cfg.MAX_PADDING # 375

class Playlist_Model(tf.keras.Model):
    def __init__(self, layer_sizes, vocab_dict):
        super().__init__()

        # ========================================
        # non-sequence playlist features
        # ========================================
        
        # Feature: playlist name
        self.pl_name_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    max_tokens=len(vocab_dict["name"]), # not needed if passing vocab
                    # vocabulary=vocab_dict['name'], 
                    name="pl_name_txt_vectorizer", 
                    ngrams=2
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["name"]) + 1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="pl_name_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="pl_name_pooling"),
            ], name="pl_name_emb_model"
        )
        
        # Feature: collaborative
        collaborative_vocab = np.array([b'false', b'true'])
        
        self.pl_collaborative_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=collaborative_vocab, 
                    mask_token=None, 
                    name="pl_collaborative_lookup", 
                    output_mode='int'
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(collaborative_vocab) + 1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="pl_collaborative_emb_layer",
                ),
            ], name="pl_collaborative_emb_model"
        )
        
        # Feature: pid
        self.pl_pid_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=tf.constant(vocab_dict['unique_pids']), 
                    mask_token=None, 
                    name="pl_pid_lookup", 
                    output_mode='int'
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict['unique_pids']),
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="pl_pid_emb_layer",
                ),
            ], name="pl_pid_emb_model"
        )
        # JW Change 9/21/2022
        # self.pl_track_uri_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.StringLookup(
        #             vocabulary=tf.constant(vocab_dict['track_uri_can']), 
        #             mask_token=None, 
        #             name="pl_track_uri_lookup", 
        #         ),
        #         tf.keras.layers.Embedding(
        #             input_dim=len(vocab_dict['track_uri_can'])+1,
        #             output_dim=EMBEDDING_DIM,
        #             mask_zero=False,
        #             name="pl_track_uri_layer",
        #         ),
        #     ], name="pl_track_uri_emb_model"
        # )

        # Feature: description_pl
        # self.pl_description_text_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.TextVectorization(
        #             max_tokens=len(vocab_dict["description_pl"]), # not needed if passing vocab
        #             # vocabulary=tf.constant(vocab_dict['description_pl']), 
        #             name="description_pl_vectorizer", 
        #             ngrams=2,
        #         ),
        #         tf.keras.layers.Embedding(
        #             input_dim=len(vocab_dict["description_pl"]) + 1,
        #             output_dim=EMBEDDING_DIM,
        #             mask_zero=False,
        #             name="description_pl_emb_layer",
        #         ),
        #         tf.keras.layers.GlobalAveragePooling1D(name="description_pl_pooling"),
        #     ], name="pl_description_emb_model"
        # )
        
        # Feature: duration_ms_seed_pl                      
        # TODO: Noramlize or Descritize?
        duration_ms_seed_pl_buckets = np.linspace(
            vocab_dict['min_duration_ms_seed_pl'], 
            vocab_dict['max_duration_ms_seed_pl'], 
            num=1000
        )
        self.duration_ms_seed_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(duration_ms_seed_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(duration_ms_seed_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM, 
                    name="duration_ms_seed_pl_emb_layer",
                )
            ], name="duration_ms_seed_pl_emb_model"
        )
        # self.duration_ms_seed_pl_normalization = tf.keras.layers.Normalization(
        #     mean=vocab_dict['avg_duration_ms_seed_pl'],
        #     variance=vocab_dict['var_duration_ms_seed_pl'],
        #     axis=None
        # )
        
        # Feature: n_songs_pl
        # TODO: Noramlize or Descritize?
        n_songs_pl_buckets = np.linspace(
            vocab_dict['min_n_songs_pl'], 
            vocab_dict['max_n_songs_pl'], 
            num=100
        )
        self.n_songs_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(n_songs_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(n_songs_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM, 
                    name="n_songs_pl_emb_layer",
                )
            ], name="n_songs_pl_emb_model"
        )
        # self.n_songs_pl_normalization = tf.keras.layers.Normalization(
        #     mean=vocab_dict['avg_n_songs_pl'],
        #     variance=vocab_dict['var_n_songs_pl'],
        #     axis=None
        # )
        
        # Feature: num_artists_pl
        # TODO: Noramlize or Descritize?
        n_artists_pl_buckets = np.linspace(
            vocab_dict['min_n_artists_pl'], 
            vocab_dict['max_n_artists_pl'], 
            num=100
        )
        self.n_artists_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(n_artists_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(n_artists_pl_buckets) + 2, 
                    output_dim=EMBEDDING_DIM, 
                    name="n_artists_pl_emb_layer",
                    mask_zero=True
                )
            ], name="n_artists_pl_emb_model"
        )
        # self.n_artists_pl_normalization = tf.keras.layers.Normalization(
        #     mean=vocab_dict['avg_n_artists_pl'],
        #     variance=vocab_dict['var_n_artists_pl'],
        #     axis=None
        # )
        
        # Feature: num_albums_pl
        n_albums_pl_buckets = np.linspace(
            vocab_dict['min_n_albums_pl'], 
            vocab_dict['max_n_albums_pl'],
            num=100
        )
        self.n_albums_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(n_albums_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(n_albums_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM, 
                    name="n_albums_pl_emb_layer",
                )
            ], name="n_albums_pl_emb_model"
        )
        # self.n_albums_pl_normalization = tf.keras.layers.Normalization(
        #     mean=vocab_dict['avg_n_albums_pl'],
        #     variance=vocab_dict['var_n_albums_pl'],
        #     axis=None
        # )
        
        # ========================================
        # sequence playlist features
        # ========================================
        
        # Feature: artist_name_pl
        self.artist_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict['artist_name_pl'], mask_token=''),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict['artist_name_pl']) + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_name_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="artist_name_pl_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="artist_name_gru"),
            ], name="artist_name_pl_emb_model"
        )
        
        # Feature: track_uri_pl
        # 2.2M unique
        self.track_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict['track_uri_pl']) + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_uri_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="track_uri_gru"),
            ], name="track_uri_pl_emb_model"
        )
        
        # Feature: track_name_pl
        self.track_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict['track_name_pl'], 
                    name="track_name_pl_lookup",
                    output_mode='int',
                    mask_token=''
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict['track_name_pl']) + 2, 
                    output_dim=EMBEDDING_DIM,
                    name="track_name_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_name_pl_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="track_name_gru"),
            ], name="track_name_pl_emb_model"
        )
        
        Feature: duration_ms_songs_pl
        duration_ms_songs_pl_buckets = np.linspace(
            vocab_dict['min_duration_ms_songs_pl'], 
            vocab_dict['max_duration_ms_songs_pl'], 
            num=100
        )
        self.duration_ms_songs_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Discretization(duration_ms_songs_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(duration_ms_songs_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="duration_ms_songs_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="duration_ms_songs_pl_emb_layer_pl_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="duration_ms_songs_gru"),
            ], name="duration_ms_songs_pl_emb_model"
        )
        
        # Feature: album_name_pl
        self.album_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.StringLookup(
                    vocabulary=tf.constant(vocab_dict['album_name_pl']), mask_token='', name="album_name_pl_lookup"),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict['album_name_pl']) + 2, 
                    output_dim=EMBEDDING_DIM,
                    name="album_name_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="album_name_pl_emb_layer_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="album_name_gru"),
            ], name="album_name_pl_emb_model"
        )
        
        # Feature: artist_pop_pl
        artist_pop_pl_buckets = np.linspace(
            vocab_dict['min_artist_pop'], 
            vocab_dict['max_artist_pop'], 
            num=10
        )
        self.artist_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Discretization(artist_pop_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(artist_pop_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_pop_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(EMBEDDING_DIM, name="artist_pop_conv1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="artist_pop_gru"),
            ], name="artist_pop_pl_emb_model"
        )
        
        # Feature: artists_followers_pl
        artists_followers_pl_buckets = np.linspace(
            vocab_dict['min_artist_followers'], 
            vocab_dict['max_artist_followers'], 
            num=10
        )
        self.artists_followers_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Discretization(artists_followers_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(artists_followers_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artists_followers_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="artists_followers_pl_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="artist_followers_gru"),
            ], name="artists_followers_pl_emb_model"
        )
        
        # Feature: track_pop_pl
        track_pop_pl_buckets = np.linspace(
            vocab_dict['min_track_pop'], 
            vocab_dict['max_track_pop'], 
            num=10
        )
        self.track_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(dtype=tf.float32),
                tf.keras.layers.Discretization(track_pop_pl_buckets.tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(track_pop_pl_buckets) + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_pop_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_pop_pl_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="track_pop_gru"),
            ], name="track_pop_pl_emb_model"
        )
        
        # Feature: artist_genres_pl
        self.artist_genres_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict['artist_genres_pl'], mask_token=''),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict['artist_genres_pl'])+2, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_genres_pl_emb_layer",
                    mask_zero=False
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_pl_1d"),
                tf.keras.layers.GRU(EMBEDDING_DIM, name="artist_genres_gru"),
            ], name="artist_genres_pl_emb_model"
        )
        
        # ========================================
        # dense and cross layers
        # ========================================

        # Cross Layers
        if USE_CROSS_LAYER:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=PROJECTION_DIM,
                kernel_initializer="glorot_uniform", 
                name="pl_cross_layer"
            )
        else:
            self._cross_layer = None
            
        # Dense Layers
        self.dense_layers = tf.keras.Sequential(name="pl_dense_layers")
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=initializer,
                )
            )
            if DROPOUT:
                self.dense_layers.add(tf.keras.layers.Dropout(DROPOUT_RATE))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=initializer
                )
            )
        ### ADDING L2 NORM AT THE END
        self.dense_layers.add(
            tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(
                    x, 1, epsilon=1e-12, name="normalize_dense"
                )
            )
        )
        
    # ========================================
    # call
    # ========================================
    def call(self, data):
        '''
        The call method defines what happens when
        the model is called
        '''
        
        all_embs = tf.concat(
            [
                self.pl_name_text_embedding(data['name']),
                self.pl_collaborative_embedding(data['collaborative']),
                self.pl_pid_embedding(data["pid"]),
                # self.pl_track_uri_embedding(data["track_uri_can"]),
                # self.pl_description_text_embedding(data['description_pl']),
                self.duration_ms_seed_pl_embedding(data["duration_ms_seed_pl"]),
                # tf.reshape(self.duration_ms_seed_pl_normalization(data["duration_ms_seed_pl"]), (-1, 1))      # Normalize or Discretize?
                self.n_songs_pl_embedding(data["n_songs_pl"]),
                # tf.reshape(self.n_songs_pl_normalization(data["n_songs_pl"]), (-1, 1))                        # Normalize or Discretize?
                self.n_artists_pl_embedding(data['num_artists_pl']),
                # tf.reshape(self.n_artists_pl_normalization(data["num_artists_pl"]), (-1, 1))                  # Normalize or Discretize?
                self.n_albums_pl_embedding(data["num_albums_pl"]),
                # tf.reshape(self.n_albums_pl_normalization(data["num_albums_pl"]), (-1, 1))                    # Normalize or Discretize?
                
                # sequence features
                # data["pos_pl"],
                self.artist_name_pl_embedding(data["artist_name_pl"]),
                self.track_uri_pl_embedding(data["track_uri_pl"]),
                self.track_name_pl_embedding(data["track_name_pl"]),
                self.duration_ms_songs_pl_embedding(data["duration_ms_songs_pl"]),
                self.album_name_pl_embedding(data["album_name_pl"]),
                self.artist_pop_pl_embedding(data["artist_pop_pl"]),
                self.artists_followers_pl_embedding(data["artists_followers_pl"]),
                self.track_pop_pl_embedding(data["track_pop_pl"]),
                self.artist_genres_pl_embedding(data["artist_genres_pl"]),
            ], axis=1)
        
                # JW 9/22/22
                # self.artist_name_pl_embedding(tf.reshape(data["artist_name_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.track_uri_pl_embedding(tf.reshape(data["track_uri_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.track_name_pl_embedding(tf.reshape(data["track_name_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.duration_ms_songs_pl_embedding(tf.reshape(data["duration_ms_songs_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.album_name_pl_embedding(tf.reshape(data["album_name_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.artist_pop_pl_embedding(tf.reshape(data["artist_pop_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.artists_followers_pl_embedding(tf.reshape(data["artists_followers_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.track_pop_pl_embedding(tf.reshape(data["track_pop_pl"], (-1, MAX_PLAYLIST_LENGTH))),
                # self.artist_genres_pl_embedding(tf.reshape(data["artist_genres_pl"], (-1, MAX_PLAYLIST_LENGTH))),
        
        # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)

# ====================================================
# Track (candidate) Tower
# ====================================================
class Candidate_Track_Model(tf.keras.Model):
    def __init__(self, layer_sizes, vocab_dict):
        super().__init__()
        
        # ========================================
        # Candidate features
        # ========================================
        
        # Feature: artist_name_can
        self.artist_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    vocabulary=vocab_dict["artist_name_can"],
                    name="artist_name_can_txt_vectorizer",
                    ngrams=2,
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["artist_name_can"]) + 1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="artist_name_can_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_name_can_pooling"),
            ], name="artist_name_can_emb_model"
        )
        
        # Feature: track_name_can
        self.track_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    vocabulary=vocab_dict["track_name_can"],
                    name="track_name_can_txt_vectorizer",
                    ngrams=2,
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["track_name_can"]) + 1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="track_name_can_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_name_can_pooling"),
            ], name="track_name_can_emb_model"
        )
        
        # Feature: album_name_can
        self.album_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    vocabulary=vocab_dict["album_name_can"],
                    name="album_name_can_txt_vectorizer",
                    ngrams=2,
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["album_name_can"]) + 1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="album_name_can_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_name_can_pooling"),
            ], name="album_name_can_emb_model"
        )
        
        # Feature: artist_uri_can
        self.artist_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim = len(vocab_dict['artist_uri_can'])+1, # TODO: JW - 200_000 + 1, 
                    output_dim = EMBEDDING_DIM,
                    name="artist_uri_can_emb_layer",
                ),
            ], name="artist_uri_can_emb_model"
        )
        
        # Feature: track_uri_can
        self.track_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim = len(vocab_dict['track_uri_can'])+1, # TODO: JW - 200_000 + 1,  
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_can_emb_layer",
                ),
            ], name="track_uri_can_emb_model"
        )
        
        # Feature: album_uri_can
        self.album_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim = len(vocab_dict['album_uri_can'])+1, # TODO: JW - 200_000 + 1,  
                    output_dim=EMBEDDING_DIM,
                    name="album_uri_can_emb_layer",
                ),
            ], name="album_uri_can_emb_model"
        )
        
        # Feature: duration_ms_can
        self.duration_ms_can_normalized = tf.keras.layers.Normalization(
            mean=vocab_dict['avg_duration_ms_songs_pl'],
            variance=vocab_dict['var_duration_ms_songs_pl'],
            axis=None
        )
        
        # Feature: track_pop_can
        self.track_pop_can_normalized = tf.keras.layers.Normalization(
            mean=vocab_dict['avg_track_pop'],
            variance=vocab_dict['var_track_pop'],
            axis=None
        )
        
        # Feature: artist_pop_can
        self.artist_pop_can_normalized = tf.keras.layers.Normalization(
            mean=vocab_dict['avg_artist_pop'],
            variance=vocab_dict['var_artist_pop'],
            axis=None
        )
        
        # Feature: artist_followers_can
        self.artist_followers_can_normalized = tf.keras.layers.Normalization(
            mean=vocab_dict['avg_artist_followers'],
            variance=vocab_dict['var_artist_followers'],
            axis=None
        )
        
        # Feature: artist_genres_can
        self.artist_genres_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    vocabulary=vocab_dict["artist_genres_can"],
                    name="artist_genres_can_txt_vectorizer",
                    ngrams=2,
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["artist_genres_can"])+1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="artist_genres_can_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_can_pooling"),
            ], name="artist_genres_can_emb_model"
        )
        
        # ========================================
        # Dense & Cross Layers
        # ========================================
        
        # Cross Layers
        if USE_CROSS_LAYER:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=PROJECTION_DIM,
                kernel_initializer="glorot_uniform", 
                name="can_cross_layer"
            )
        else:
            self._cross_layer = None
        
        # Dense Layer
        self.dense_layers = tf.keras.Sequential(name="candidate_dense_layers")
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=initializer,
                )
            )
            if DROPOUT:
                self.dense_layers.add(tf.keras.layers.Dropout(DROPOUT_RATE))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=initializer
                )
            )
            
    # ========================================
    # Call Function
    # ========================================
            
    def call(self, data):
        
        all_embs = tf.concat(
            [
                self.artist_name_can_text_embedding(data['artist_name_can']),
                self.track_name_can_text_embedding(data['track_name_can']),
                self.album_name_can_text_embedding(data['album_name_can']),
                self.artist_uri_can_embedding(data['artist_uri_can']),
                self.track_uri_can_embedding(data['track_uri_can']),
                self.album_uri_can_embedding(data['album_uri_can']),
                tf.reshape(self.duration_ms_can_normalized(data["duration_ms_can"]), (-1, 1)),
                tf.reshape(self.track_pop_can_normalized(data["track_pop_can"]), (-1, 1)),
                tf.reshape(self.artist_pop_can_normalized(data["artist_pop_can"]), (-1, 1)),
                tf.reshape(self.artist_followers_can_normalized(data["artist_followers_can"]), (-1, 1)),
                self.artist_genres_can_text_embedding(data['album_uri_can']),
            ], axis=1
        )
        
        # return self.dense_layers(all_embs)
                # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)

# ====================================================
# Combined 2Tower
# ====================================================
class TheTwoTowers(tfrs.models.Model):

    def __init__(self, layer_sizes, vocab_dict_load, parsed_candidate_dataset):
        super().__init__()
        
        self.query_tower = Playlist_Model(layer_sizes, vocab_dict_load)
        
        self.candidate_tower = Candidate_Track_Model(layer_sizes, vocab_dict_load)
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidate_dataset.batch(128).cache().map(self.candidate_tower) # TODO: parameterize
            )
        )
        
    def compute_loss(self, data, training=False):
        query_embeddings = self.query_tower(data)
        candidate_embeddings = self.candidate_tower(data)

        return self.task(
            query_embeddings, 
            candidate_embeddings, 
            compute_metrics=not training
        ) # turn off metrics to save time on training
