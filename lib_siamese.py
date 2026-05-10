"""Siamese Network (MLP backbone) — Tầng 2.

Lưu ý kỹ thuật quan trọng:
  KHÔNG dùng @st.cache_resource cho hàm build_siamese_model.

  Lý do: cache_resource giữ lại MỘT instance duy nhất của model giữa các lần
  rerun của Streamlit. Nếu user train lần thứ 2, model đã có sẵn trọng số từ
  lần trước → model.fit() tiếp tục từ trạng thái cũ → có thể bị stuck ở local
  minimum xấu (AUC < 0.5 do final Dense học sai dấu cho khoảng cách Euclid).

  Fix: clear_session() + build mới mỗi lần gọi → đảm bảo training reproducible.
"""


def build_siamese_model():
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K

    # Reset graph + clear cache để training luôn xuất phát từ trọng số mới
    K.clear_session()
    tf.random.set_seed(42)

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
    # Khởi tạo Dense cuối với BIAS lớn dương + KERNEL âm để score~1 khi distance~0
    # Việc này giúp model hội tụ đúng hướng ngay từ đầu (không bị "anti-learning")
    out = Dense(
        1, activation='sigmoid', name="similarity_score",
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.5, -0.05),
        bias_initializer=tf.keras.initializers.Constant(2.0),
    )(dist)
    model = Model([in1, in2], out, name="siamese_network")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, mlp
