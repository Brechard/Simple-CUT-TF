import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

KERNEL_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
GAMMA_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def get_filter(filt_size):
    filter_ = np.array([1., ])
    if filt_size == 2:
        filter_ = np.array([1., 1.])
    elif filt_size == 3:
        filter_ = np.array([1., 2., 1.])
    elif filt_size == 4:
        filter_ = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        filter_ = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        filter_ = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        filter_ = np.array([1., 6., 15., 20., 15., 6., 1.])
    filter_ = filter_[:, None] * filter_[None, :]
    filter_ = filter_ / np.sum(filter_)
    return filter_


class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


class BlurPool(tf.keras.layers.Layer):
    def __init__(self, filt_size=3, stride=2):
        super(BlurPool, self).__init__()
        self.strides = (stride, stride)
        self.filt_size = filt_size

        self.filter = get_filter(filt_size)
        self.pad_layer = ReflectionPadding2D()

    def compute_output_shape(self, input_shape):
        height = input_shape[1] // self.strides[0]
        width = input_shape[2] // self.strides[1]
        channels = input_shape[3]
        return input_shape[0], height, width, channels

    def call(self, x):
        filter_ = self.filter
        filter_ = np.tile(filter_[:, :, None, None], (1, 1, tf.keras.backend.int_shape(x)[-1], 1))
        filter_ = tf.keras.backend.constant(filter_, dtype=tf.keras.backend.floatx())
        x = self.pad_layer(x)
        x = tf.keras.backend.depthwise_conv2d(x, filter_, strides=self.strides, padding='valid')
        return x


class ReplicationPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode='SYMMETRIC')


class Upsample(tf.keras.layers.Layer):
    def __init__(self, batch_size, filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.strides = (stride, stride)
        self.off = int((stride - 1) / 2.)
        self.batch_size = batch_size

        self.filter = get_filter(filt_size=self.filt_size) * (stride ** 2)

    def compute_output_shape(self, input_shape):
        height = input_shape[1] * self.strides[0]
        width = input_shape[2] * self.strides[1]
        channels = input_shape[3]
        return self.batch_size, height, width, channels

    def call(self, x):
        filter_ = self.filter
        filter_ = np.tile(filter_[:, :, None, None], (1, 1, 1, tf.keras.backend.int_shape(x)[-1]))
        filter_ = tf.keras.backend.constant(filter_, dtype=tf.keras.backend.floatx())
        ret_val = tf.nn.conv2d_transpose(x, filter_,
                                         output_shape=self.compute_output_shape(tf.keras.backend.int_shape(x)),
                                         strides=self.strides)
        return ret_val


def residual_block(input_tensor, activation, kernel_initializer=KERNEL_INIT, kernel_size=(3, 3), strides=(1, 1),
                   padding="valid", gamma_initializer=GAMMA_INIT, use_bias=False, res_block_n=None):
    dim = input_tensor.shape[-1]
    input_tensor = layers.Input(input_tensor.shape[1:])

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return tf.keras.models.Model(input_tensor, x, name=f'residual_block_{res_block_n}')


def downsample(input_tensor, filters, activation, kernel_initializer=KERNEL_INIT, kernel_size=(3, 3),
               use_antialias=None, padding="same", gamma_initializer=GAMMA_INIT, use_bias=False):
    strides = (1, 1) if use_antialias or use_antialias is None else (2, 2)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(input_tensor)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    if use_antialias:
        x = BlurPool()(x)
    return x


def upsample(input_tensor, batch_size, filters, activation, kernel_size=(3, 3), use_antialias=True, padding="same",
             kernel_initializer=KERNEL_INIT, gamma_initializer=GAMMA_INIT, use_bias=False):
    if use_antialias:
        x = Upsample(batch_size)(input_tensor)
    strides = (1, 1) if use_antialias else (2, 2)
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_encoder(image_size, filters=16, num_downsampling_blocks=2, num_residual_blocks=4, use_antialias=True,
                       gamma_initializer=GAMMA_INIT, name='Encoder'):
    img_input = layers.Input(shape=image_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=KERNEL_INIT, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for i in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"), use_antialias=use_antialias)

    # Residual blocks
    for i in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"), res_block_n=i)(x)

    return tf.keras.models.Model(img_input, x, name=name)


def get_resnet_decoder(batch_size, input_shape, filters=64, num_upsample_blocks=2, use_antialias=True, name='Decoder'):
    img_input = layers.Input(shape=input_shape, name=name + "_img_input")
    x = img_input
    # Upsampling
    for i in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, batch_size, filters, activation=layers.Activation("relu"), use_antialias=use_antialias)

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    return tf.keras.models.Model(img_input, x, name=name)


def get_generator(image_size, batch_size, use_antialias, num_downsampling_blocks=2, num_residual_blocks=4,
                  num_upsample_blocks=2):
    encoder = get_resnet_encoder(image_size,
                                 use_antialias=use_antialias,
                                 num_downsampling_blocks=num_downsampling_blocks,
                                 num_residual_blocks=num_residual_blocks)
    decoder = get_resnet_decoder(batch_size, encoder.output.shape[1:],
                                 use_antialias=use_antialias,
                                 num_upsample_blocks=num_upsample_blocks)

    img_input = layers.Input(shape=image_size, name="generator_img_input")
    x = img_input
    x = encoder(x)
    x = decoder(x)
    return tf.keras.models.Model(img_input, x, name='Generator')


def get_discriminator(image_size, filters=16, use_antialias=True, kernel_initializer=KERNEL_INIT, num_downsampling=3):
    img_input = layers.Input(shape=image_size, name="discriminator_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(num_downsampling):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                use_antialias=use_antialias
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                use_antialias=None
            )

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

    model = tf.keras.models.Model(inputs=img_input, outputs=x, name='Discriminator')
    return model


def get_features(encoder, x):
    """ Extract the features generated by each block of the encoder model """
    x = encoder.layers[0](x)  # This is just the input layer
    features = []
    for layer in encoder.layers[1:]:
        x = layer(x)
        features.append(x)
    return features


def normalize(x):
    norm = tf.pow(tf.reduce_sum(tf.math.pow(x, 2), axis=1, keepdims=True), 1 / 2)
    out = tf.divide(x, norm + 1e-7)
    return out


class MLP:
    def __init__(self, dimension=256, num_patches=64):
        self.dim = dimension
        self.num_patches = num_patches
        self.n_mlps = 0

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            mlp = tf.keras.Sequential([tf.keras.layers.Dense(self.dim, input_dim=feat.shape[-1]),
                                       tf.keras.layers.ReLU(),
                                       tf.keras.layers.Dense(self.dim)])
            setattr(self, f'mlp_{mlp_id}', mlp)
        self.n_mlps = mlp_id + 1

    def forward(self, features, patch_ids=None, use_mlp=True):
        """
        Forward the features through their corresponding Multi Layer Perceptron.
        If the Patch IDs are not provided it means that it is the first time being used with this Batch. What we do
        then is randomly select "num_patches" of patches to process and return the IDs so that in the second run we
        execute the same patches.
        Args:
            features: Features extracted from the encoder. Shape must be [Batch size, Heigh, Width, Channels]
            patch_ids: IDs of the paths to execute. If None, they will be randomly chosen.

        Returns:
            Processed features and the patch ids.
        """
        if self.n_mlps == 0 and use_mlp:
            self.create_mlp(features)
        return_ids, return_feats = [], []
        for feat_id, feat in enumerate(features):
            B, H, W, C = feat.shape
            feat_reshape = tf.reshape(feat, (B, -1, C))

            if patch_ids is None:
                patch_id = tf.random.shuffle(tf.range(H * W))[:self.num_patches]
            else:
                patch_id = patch_ids[feat_id]

            x_sample = tf.reshape(tf.gather(feat_reshape, patch_id, axis=1), (-1, C))  # reshape(-1, x.shape[1])
            if use_mlp:
                mlp = getattr(self, f'mlp_{feat_id}')
                x_sample = mlp(x_sample)
            #             x_sample = normalize(x_sample)
            return_ids.append(patch_id)
            return_feats.append(x_sample)

        return return_feats, return_ids


def patch_nce_loss(feat_src, feat_tgt, nce_T):
    n_patches, size = feat_src.shape

    l_pos = tf.matmul(tf.reshape(feat_src, (n_patches, 1, -1)), tf.reshape(feat_tgt, (n_patches, -1, 1)))
    l_pos = tf.squeeze(l_pos, 1)

    # reshape features to batch size

    l_neg = tf.matmul(feat_src, tf.transpose(feat_tgt))

    # diagonal entries are similarity between same features, and hence meaningless.
    # Since there is no masked_fill method in tensorflow we will multiply by a unitary matrix with almost zero values
    # in the diagonal
    diagonal = tf.ones((n_patches, n_patches)) - tf.eye(n_patches) * 0.9999999
    l_neg = l_neg * diagonal

    out = tf.concat((l_pos, l_neg), axis=1) / nce_T
    target = [[1] + [0.0] * l_neg.shape[1] for i in range(out.shape[0])]
    #     tf.print(target)
    loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)(target,
                                                                                                               out)
    #     tf.print('entropy loss:', loss)

    return loss


def gan_loss(y, target_is_real, target_real_label=1.0, target_fake_label=0.0):
    if target_is_real:
        return tf.keras.losses.MSE(y, target_real_label)

    return tf.keras.losses.MSE(y, target_fake_label)


def calculate_NCE_loss(generator, mlp, nce_T, lambda_NCE, use_mlp, src, tgt):
    input_layer, encoder = generator.layers[:2]
    encoded_features_src = get_features(encoder, input_layer(src))
    encoded_features_tgt = get_features(encoder, input_layer(tgt))

    mlp_features_src, feat_ids = mlp.forward(encoded_features_src, use_mlp=use_mlp)
    mlp_features_tgt, _ = mlp.forward(encoded_features_tgt, feat_ids, use_mlp=use_mlp)

    total_nce_loss = 0
    losses = []
    for feat_src, feat_tgt in zip(mlp_features_src, mlp_features_tgt):
        nce_loss = patch_nce_loss(feat_src, feat_tgt, nce_T) * lambda_NCE
        losses.append(tf.reduce_mean(nce_loss))
        total_nce_loss += tf.reduce_mean(nce_loss)
    return total_nce_loss / len(encoded_features_tgt), losses


class CUT(tf.keras.Model):
    def __init__(
            self,
            generator,
            discriminator,
            mlp,
            use_mlp=True,
            lambda_NCE=1.0,
            lambda_GAN=1.0,
            fast=False,
            nce_T=0.07,
            use_defaults=True
    ):
        super(CUT, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.input_layer, self.encoder = generator.layers[:2]
        self.mlp = mlp
        self.fast = fast
        self.use_mlp = True if use_defaults else use_mlp
        self.lambda_NCE = (10.0 if fast else 1.0) if use_defaults else lambda_NCE
        self.lambda_GAN = 1.0 if use_defaults else lambda_GAN
        self.nce_T = 0.07 if use_defaults else nce_T

    def compile(
            self,
            generator_optimizer,
            discriminator_optimizer,
            mlp_optimizer,
            discriminator_loss_fn
    ):
        super(CUT, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.mlp_optimizer = mlp_optimizer
        self.discriminator_loss_fn = discriminator_loss_fn

    def calculate_NCE_loss(self, source, target):
        return calculate_NCE_loss(self.generator, self.mlp, self.nce_T, self.lambda_NCE, self.use_mlp, source, target)

    def train_step(self, batch_data):
        # x is PHOTO and y is MONET
        source, target = batch_data

        # For FastCUT, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass source image through the generator to calculate the fake target image.
        # 2. Call the discriminator with the fake target and real target.
        # 3. Calculate the generator loss (adversarial + NCE).
        # 4. Calculate the discriminator loss.
        # 5. Update the weights of the generators
        # 6. Update the weights of the discriminators
        # 7. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Photo to fake Monet
            fake_target = self.generator(source, training=True)

            pred_fake = self.discriminator(fake_target, training=True)
            pred_real = self.discriminator(target, training=True)

            ###########################
            ##### TRAIN GENERATOR #####
            ###########################

            # First, G(photo) should fake the discriminator
            loss_G_GAN = tf.reduce_mean(
                self.discriminator_loss_fn(pred_fake, True)) * self.lambda_GAN if self.lambda_GAN > 0 else 0

            total_loss_nce, losses = loss_nce, losses = self.calculate_NCE_loss(source,
                                                                                fake_target) if self.lambda_NCE > 0 else 0
            loss_nce_identity, losses_identity = 0, [0 for i in losses]
            if not self.fast and self.lambda_NCE > 0:
                fake_identity = self.generator(target, training=True)
                loss_nce_identity, losses_identity = self.calculate_NCE_loss(target, fake_identity)
                total_loss_nce = (total_loss_nce + loss_nce_identity) / 2

            loss_G = loss_G_GAN + total_loss_nce

            ###########################
            ### TRAIN DISCRIMINATOR ###
            ###########################

            loss_D_fake = tf.reduce_mean(self.discriminator_loss_fn(pred_fake, False))
            loss_D_real = tf.reduce_mean(self.discriminator_loss_fn(pred_real, True))
            loss_D = (loss_D_fake + loss_D_real) / 2

        # Get the gradients for the generators
        generator_gradients = tape.gradient(loss_G, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(loss_D, self.discriminator.trainable_variables)
        if self.use_mlp:
            mlp_gradients = []
            for feat_id in range(self.mlp.n_mlps):
                mlp = getattr(self.mlp, f'mlp_{feat_id}')
                mlp_gradients = tape.gradient(loss_G, mlp.trainable_variables)
                self.mlp_optimizer.apply_gradients(zip(mlp_gradients, mlp.trainable_variables))

        # Update the weights
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        return {
            "G_loss": loss_G,
            "D_loss": loss_D,
            "loss_G_GAN": loss_G_GAN,
            "total_loss_nce": total_loss_nce,
            "loss_nce": loss_nce,
            "loss_nce_idt": loss_nce_identity,
            'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real
        }


class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, generator, source_ds, target_ds, out_dir, num_img=2, every_n_epoch=5):
        self.num_img = num_img
        self.generator = generator
        self.every_n_epoch = every_n_epoch
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_n_epoch != 0:
            return
        _, ax = plt.subplots(self.num_img, 4, figsize=(20, 10))
        [ax[0, i].set_title(title) for i, title in enumerate(["Source", "Fake target", "Target", "Identity target"])]
        for i, (source, target) in enumerate(zip(self.source_ds.take(self.num_img), self.target_ds.take(self.num_img))):
            prediction = self.generator(source)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            source = (source[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            idt_target = self.generator(target)[0].numpy()
            idt_target = (idt_target * 127.5 + 127.5).astype(np.uint8)
            target = (target[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(source)
            ax[i, 1].imshow(prediction)
            ax[i, 2].imshow(target)
            ax[i, 3].imshow(idt_target)

            [ax[i, j].axis("off") for j in range(4)]

        plt.savefig(f'epoch={epoch + 1}.png')
        plt.show()
        plt.close()


def plot_history_losses(history, suptitle):
    n_cols = 4
    n_rows = len(history) // n_cols
    if len(history) % n_cols != 0:
        n_rows += 1
    fig = plt.figure(figsize=(20, 10))
    i = 1
    for key, items in history.items():
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.plot(items)
        ax.set_title(key)
        i += 1
    plt.suptitle(suptitle)
    plt.show()
