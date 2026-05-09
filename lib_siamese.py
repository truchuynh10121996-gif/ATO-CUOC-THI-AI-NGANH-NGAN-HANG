"""Siamese Network (MLP backbone) — Tầng 2."""
import streamlit as st


@st.cache_resource
def build_siamese_model():
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model

    def build_mlp():
        inp = Input(shape=(12,))
        x = Dense(64, activation='relu')(inp)
        x = Dense(32, activation='relu')(x)
        x = Dense(16)(x)
        return Model(inp, x, name="shared_mlp")

    mlp = build_mlp()
    in1 = Input(shape=(12,), name="session_A")
    in2 = Input(shape=(12,), name="session_B")
    e1, e2 = mlp(in1), mlp(in2)
    dist = Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True) + 1e-8),
        name="euclidean_distance",
    )([e1, e2])
    out = Dense(1, activation='sigmoid', name="similarity_score")(dist)
    model = Model([in1, in2], out, name="siamese_network")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, mlp
