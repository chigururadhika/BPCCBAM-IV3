import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionV3
import tensorflow.keras.backend as K

def channel_attention(input_tensor, reduction=16, name="CAM"):
    channel = input_tensor.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    max_pool = layers.GlobalMaxPooling2D()(input_tensor)
    shared = layers.Dense(channel // reduction, activation='relu', name=name + "_mlp_1")
    shared_out = layers.Dense(channel, name=name + "_mlp_2")
    avg_out = shared_out(shared(avg_pool))
    max_out = shared_out(shared(max_pool))
    out = layers.Add()([avg_out, max_out])
    out = layers.Activation('sigmoid')(out)
    out = layers.Reshape((1, 1, channel))(out)
    return layers.Multiply()([input_tensor, out])


class BDASAM(layers.Layer):

    def __init__(self, num_bits=8, use_soft_bdp=False, name="BDASAM"):
        super().__init__(name=name)
        self.num_bits = num_bits
        self.use_soft_bdp = use_soft_bdp
        self.conv = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid', name=name + "_conv")

    def call(self, features):

        intensity = tf.reduce_mean(features, axis=-1, keepdims=True)  # [B,H,W,1]

        if self.use_soft_bdp:

            scaled = (intensity - tf.reduce_min(intensity)) / (tf.reduce_max(intensity) - tf.reduce_min(intensity) + 1e-8)
            scaled = scaled * 255.0
            bit_maps = []
            for b in range(self.num_bits):
                thresh = (2 ** b)
               
                smooth = tf.sigmoid((scaled - thresh) / 3.0)  

                bit = smooth * (1.0 - tf.sigmoid((scaled - (thresh + 1.0)) / 3.0))
                bit_maps.append(bit)
            bit_stack = tf.concat(bit_maps, axis=-1) 
        else:

            minv = tf.reduce_min(intensity)
            maxv = tf.reduce_max(intensity)
            scaled = (intensity - minv) / (maxv - minv + 1e-8)
            scaled = tf.cast(tf.clip_by_value(tf.round(scaled * 255.0), 0, 255), tf.uint8)  # [B,H,W,1]

            scaled8 = tf.squeeze(scaled, axis=-1)  

            bit_maps = []
            for b in range(self.num_bits):
                mask = tf.bitwise.bitwise_and(scaled8, tf.constant(1 << b, dtype=tf.uint8))
                bit = tf.cast(tf.not_equal(mask, 0), tf.float32)
                bit = tf.expand_dims(bit, axis=-1)  
                bit_maps.append(bit)
            bit_stack = tf.concat(bit_maps, axis=-1) 

        attention = self.conv(bit_stack) 

        out = attention * features
        return out



def position_attention_module(input_tensor, name="PAM"):

    _, h, w, c = input_tensor.shape
    reduced_c = max(1, c // 8)

    f = layers.Conv2D(reduced_c, 1, padding='same', name=name + "_f")(input_tensor) 
    g = layers.Conv2D(reduced_c, 1, padding='same', name=name + "_g")(input_tensor) 
    h_conv = layers.Conv2D(c, 1, padding='same', name=name + "_h")(input_tensor)    

    def hw_flat(x):
        b = tf.shape(x)[0]
        x_resh = tf.reshape(x, [b, -1, x.shape[-1]]) 
        return x_resh

    f_flat = hw_flat(f) 
    g_flat = hw_flat(g)   
    h_flat = hw_flat(h_conv) 

    S = tf.matmul(f_flat, g_flat, transpose_b=True) 
    S = tf.nn.softmax(S, axis=-1)

    out_flat = tf.matmul(S, h_flat)

    out = tf.reshape(out_flat, tf.shape(input_tensor))  
    out = layers.Conv2D(c, 1, padding='same', name=name + "_out_conv")(out)

    gamma = tf.Variable(0.0, trainable=True, dtype=tf.float32, name=name + "_gamma")
    return layers.Add()([input_tensor, gamma * out])


def create_inception_with_attentions(input_shape=(299, 299, 3),
                                     num_classes=10,
                                     freeze_backbone=True,
                                     bdasam_bits=8,
                                     bdasam_soft=True):
    """
    Build InceptionV3 + CAM + BDASAM + PAM fused model.

    Args:
      input_shape: image input shape (InceptionV3 default is 299x299x3)
      num_classes: number of output classes
      freeze_backbone: whether to freeze pretrained InceptionV3
      bdasam_bits: number of bit-planes to use
      bdasam_soft: whether BDASAM uses smooth (differentiable) bit-plane approx
    """
    inputs = layers.Input(shape=input_shape)


    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)


    backbone = InceptionV3(include_top=False, weights='imagenet', input_tensor=x)
    if freeze_backbone:
        backbone.trainable = False

    features = backbone.output  


    cam_out = channel_attention(features, reduction=16, name="CAM")


    bdasam_layer = BDASAM(num_bits=bdasam_bits, use_soft_bdp=bdasam_soft, name="BDASAM")
    bdasam_out = bdasam_layer(cam_out)

    pam_out = position_attention_module(bdasam_out, name="PAM")

    fused = layers.Add()([features, cam_out, bdasam_out, pam_out])
    fused = layers.Conv2D(512, (1, 1), padding='same', activation='relu', name="fusion_conv")(fused)
    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Dropout(0.5)(fused)
    outputs = layers.Dense(num_classes, activation='softmax', name="pred")(fused)

    model = Model(inputs, outputs, name="InceptionV3_CAM_BDASAM_PAM")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



model = create_inception_with_attentions(input_shape=(299, 299, 3),
                                            num_classes=10,
                                            freeze_backbone=True,
                                            bdasam_bits=8,
                                            bdasam_soft=True) 
model.summary()





