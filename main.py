import argparse
from collections import defaultdict
from datetime import datetime
from os import makedirs

from tqdm import tqdm

from data_utils import load_dataset
from model import *


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_keras(optimizers, generator, discriminator, mlp, use_mlp, fast, source_ds, target_ds, epochs,
                output_directory, lambda_NCE, lambda_GAN, nce_T):
    plotter = GANMonitor(generator, source_ds, target_ds, output_directory + '/images/')
    checkpoint_filepath = output_directory + "/checkpoints/epoch_{epoch:03d}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

    # Create CUT model
    model = CUT(generator=generator, discriminator=discriminator, mlp=mlp, use_mlp=use_mlp, fast=fast,
                lambda_NCE=lambda_NCE, lambda_GAN=lambda_GAN, nce_T=nce_T)

    # Compile the model
    model.compile(
        generator_optimizer=optimizers['generator'],
        discriminator_optimizer=optimizers['discriminator'],
        mlp_optimizer=optimizers['mlp'],
        discriminator_loss_fn=gan_loss
    )
    history = model.fit(
        tf.data.Dataset.zip((source_ds, target_ds)),
        epochs=epochs,
        callbacks=[plotter, model_checkpoint_callback]
    )
    return history.history


def train_tf(optimizers, generator, discriminator, mlp, use_mlp, fast, source_ds, target_ds, epochs, output_directory,
             lambda_NCE, lambda_GAN, nce_T):
    n_files = 0
    counted = False
    history = defaultdict(list)
    for epoch in range(epochs):

        # Fancy progress bar
        pbar = tqdm(tf.data.Dataset.zip((source_ds, target_ds)), total=None if not counted else n_files)

        # Batches iteration, batch_size = 1
        for source, target in pbar:
            if not counted:
                n_files += 1
                # Forward pass: needs to be recorded by gradient tape
            with tf.GradientTape(persistent=True) as tape:
                # Photo to fake Monet
                fake_target = generator(source, training=True)

                pred_fake = discriminator(fake_target, training=True)
                pred_real = discriminator(target, training=True)

                ###########################
                ##### TRAIN GENERATOR #####
                ###########################

                # First, G(photo) should fake the discriminator
                loss_G_GAN = tf.reduce_mean(gan_loss(pred_fake, True)) * lambda_GAN if lambda_GAN > 0 else 0

                total_loss_nce, losses = loss_nce, losses = calculate_NCE_loss(generator, mlp, nce_T,
                                                                               lambda_NCE, use_mlp,
                                                                               src=source, tgt=fake_target)

                loss_nce_identity, losses_identity = 0, [0 for i in losses]
                if not fast and lambda_NCE > 0:
                    fake_identity = generator(target, training=True)
                    loss_nce_identity, losses_identity = calculate_NCE_loss(generator, mlp, nce_T, lambda_NCE,
                                                                            use_mlp, src=target, tgt=fake_identity)
                    total_loss_nce = (total_loss_nce + loss_nce_identity) / 2

                loss_G = loss_G_GAN + total_loss_nce

                ###########################
                ### TRAIN DISCRIMINATOR ###
                ###########################

                loss_D_fake = tf.reduce_mean(gan_loss(pred_fake, False))
                loss_D_real = tf.reduce_mean(gan_loss(pred_real, True))
                loss_D = (loss_D_fake + loss_D_real) / 2

                history['Total Generator loss'].append(loss_G)
                history['Generator GAN loss'].append(loss_G_GAN)
                history['NCE loss'].append(loss_nce)
                if not fast and lambda_NCE > 0:
                    history['Total Generator NCE loss'].append(total_loss_nce)
                    history['Identity NCE loss'].append(loss_nce_identity)
                history['Total Discriminator loss'].append(loss_D)
                history['Discriminator fake loss'].append(loss_D_fake)
                history['Discriminator real loss'].append(loss_D_real)

            # Backward pass:
            # compute gradients w.r.t. the loss
            # update trainable weights of the model
            generator_gradients = tape.gradient(loss_G, generator.trainable_variables)
            discriminator_gradients = tape.gradient(loss_D, discriminator.trainable_variables)
            if use_mlp:
                for feat_id in range(mlp.n_mlps):
                    mlp_ = getattr(mlp, f'mlp_{feat_id}')
                    mlp_gradients = tape.gradient(loss_G, mlp_.trainable_variables)
                    optimizers['mlp'].apply_gradients(zip(mlp_gradients, mlp_.trainable_variables))

            optimizers['generator'].apply_gradients(zip(generator_gradients, generator.trainable_variables))
            optimizers['discriminator'].apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            # Tracking progress
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} G_loss {loss_G.numpy():.2f} D_loss {loss_D.numpy():.2f}, total_loss_nce {total_loss_nce:.2f}")
        counted = True

        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(source_ds.take(4)):
            prediction_ = generator(img)[0].numpy()
            prediction = (prediction_ * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

        plt.show()
    plt.close()

    return history


def train(hparams):
    [makedirs(hparams.output_directory + i, exist_ok=True) for i in ['checkpoints/', 'images/']]
    optimizers = {
        'generator': tf.keras.optimizers.Adam(learning_rate=hparams.lr, beta_1=hparams.beta_1),
        'discriminator': tf.keras.optimizers.Adam(learning_rate=hparams.lr, beta_1=hparams.beta_1),
        'mlp': tf.keras.optimizers.Adam(learning_rate=hparams.lr, beta_1=hparams.beta_1)
    }

    generator = get_generator(hparams.image_size, batch_size=hparams.batch_size, use_antialias=hparams.use_antialias,
                              num_residual_blocks=hparams.num_residual_blocks)
    discriminator = get_discriminator(hparams.image_size, use_antialias=hparams.use_antialias)
    mlp = MLP(hparams.mlp_dim, hparams.n_patches)

    fast = False if hparams.use_defaults else hparams.fast
    lambda_NCE = (10.0 if fast else 1.0) if hparams.use_defaults else hparams.lambda_NCE
    lambda_GAN = 1.0 if hparams.use_defaults else hparams.lambda_GAN
    nce_T = 0.07 if hparams.use_defaults else hparams.nce_T

    source_filenames = tf.io.gfile.glob(str(hparams.src_tfrec_path + '*.tfrec'))
    target_filenames = tf.io.gfile.glob(str(hparams.tgt_tfrec_path + '*.tfrec'))
    print(f'Found {len(source_filenames)} TFRecord Files for source and {len(target_filenames)} for target')

    source_ds = load_dataset(source_filenames, image_size=hparams.image_size, batch_size=hparams.batch_size,
                             use_augmentation=hparams.augment_src)
    target_ds = load_dataset(target_filenames, image_size=hparams.image_size, batch_size=hparams.batch_size,
                             use_augmentation=hparams.augment_tgt)

    train_fn = train_keras if hparams.keras else train_tf

    photo = [photo for photo in source_ds.take(1)][0]
    generator(photo)

    history = train_fn(optimizers, generator, discriminator, mlp, hparams.use_mlp, fast, source_ds, target_ds,
                       hparams.epochs, hparams.output_directory, lambda_NCE, lambda_GAN, nce_T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, required=False, help='directory to save checkpoints')
    parser.add_argument('--use_defaults', type=str2bool, default=True, help='Use default values')
    parser.add_argument('--use_mlp', type=str2bool, default=True, help='Use MLP for the NCE loss')
    parser.add_argument('--keras', type=str2bool, default=False, help='Use keras training (harder to debug)')
    parser.add_argument('--fast', type=str2bool, default=False, help='Use fast version of the algorithm')
    parser.add_argument('--use_antialias', type=str2bool, default=True,
                        help='Use antialiased-downsampling instead of stride = 2 (nice)')
    parser.add_argument('--lambda_NCE', type=float, default=1.0, help='Lambda value for the NCE loss')
    parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Lambda value for the GAN loss')
    parser.add_argument('--nce_T', type=float, default=0.07, help='Tau value for the NCE loss')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for optimizers')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 value for Adam optimizer')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels for the image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='Number of discriminator filters in the first conv layer')
    parser.add_argument('--mlp_dim', type=int, default=256, help='Dimension of the MLP layers')
    parser.add_argument('--n_patches', type=int, default=64, help='Number of patches for the NCE loss')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--num_residual_blocks', type=int, default=4, help='Number of residual blocks in Resnet')
    parser.add_argument('--src_tfrec_path', type=str, required=False, help='directory to tfrecords of source images')
    parser.add_argument('--tgt_tfrec_path', type=str, required=False, help='directory to tfrecords of target images')
    parser.add_argument('--augment_src', type=str2bool, default=True, help='Use data augmentation for source images')
    parser.add_argument('--augment_tgt', type=str2bool, default=True, help='Use data augmentation for target images')

    args = parser.parse_args()
    if args.output_directory is None:
        fast = False if args.use_defaults else args.fast
        args.output_directory = "./outputs/" + datetime.now().strftime("%y%m%d%H%M") + ('Fast' if fast else '') + "CUT/"
    args.image_size = (args.image_size, args.image_size, args.image_channels)
    train(args)
