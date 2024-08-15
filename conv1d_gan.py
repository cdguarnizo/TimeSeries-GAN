from numpy import zeros
from numpy import ones
import numpy as np
import pandas as pd
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D



# define the standalone discriminator model
def define_discriminator(in_shape=(384,1), n_classes=4):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=in_shape)
    
    fe = Conv1D(16, 3, strides=2, padding='same', kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    
    fe = Conv1D(32, 3, strides=2, padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    
    fe = Conv1D(64, 3, strides=2, padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    
    fe = Conv1D(128, 3, strides=2, padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
   
    fe = Flatten()(fe)
    
    # Use consistent output names
    out_validity = Dense(1, activation='sigmoid', name='validity')(fe)
    out_label = Dense(n_classes, activation='softmax', name='label')(fe)
    
    model = Model(in_image, [out_validity, out_label])
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(
        loss={'validity': 'binary_crossentropy', 'label': 'sparse_categorical_crossentropy'},
        optimizer=opt,
        loss_weights={'validity': 1.0, 'label': 1.0}
    )
    return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=4):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    depth = 32 #32
    ks = 3
    dropout = 0.25
    dim = 96 #
    # 
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 96 * 1
    li = Dense(n_nodes)(li)
    
    # reshape to additional channel
    li = Reshape((96, 1, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = dim*depth
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, 1, depth))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li]) #gen=96,1,32 x li=96,1,1
    # upsample to 192,1,16
    gen = Conv2DTranspose(16, 3, strides=(2,1), padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    #upsample to  384,1,8
    gen = Conv2DTranspose(8, 3, strides=(2,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    #updamsple
    #gen = Conv2DTranspose(48, (3,3), strides=(2,1), padding='same', kernel_initializer=init)(gen)
    #gen = BatchNormalization()(gen)
    #gen = Activation('relu')(gen)
    #384 x 1 property image
    gen = Reshape((384,-1))(gen)
    # upsample to 28x28
    #gen = Conv1DTranspose(1, 3, padding='same', kernel_initializer=init)(gen)
    gen = Conv1D(1, 3, strides=1, padding='same')(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    model.summary()
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    d_model.trainable = False
    
    gan_output = d_model(g_model.output)
    
    model = Model(g_model.input, gan_output)
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(
        loss=['binary_crossentropy','sparse_categorical_crossentropy'],
        optimizer=opt,
    )
    return model
 
# load images
def load_real_samples():
    # load dataset
    df29 = pd.read_csv('outfinaltest890.csv',header=None)
    
    #df29 = df29.iloc[1:]
    #df = df.astype('float64')
    #data11 = df29.values
    dataset=df29.values
    dataset = dataset.astype('float64')
    dataxy=dataset[:,1:]
    timep=np.zeros([len(dataset),])
    timep=dataset[:,0]
    #maxchannels=10
    maxer=np.amax(dataset[:,2])
    print (maxer)
    maxeri=maxer.astype('int')
    maxchannels=maxeri
    idataset=np.zeros([len(dataset),],dtype=int)
    idataset=dataset[:,2]
    idataset=idataset.astype(int)
    scaler = MinMaxScaler(copy=False)
    
    X_train = dataset[:,1]
    y_train = idataset[:]
    #(X_train, y_train), (_, _) = mnist.load_data()
    window=384
    n = ((np.where(np.any(dataxy, axis=1))[0][-1] + 1) // window) * window
    
    xx = scaler.fit_transform(dataxy[:n,0].reshape(-1,1))
    y_train = dataxy[:(n-window),1].reshape(-1,1)
    
    #make to matrix
    X_train = np.asarray([xx[i:i+window] for i in range (n - window)])
    #y_train = np.asarray([y_train[i:i+window] for i in range (n - window)])
    #trainX=X_train.copy()
    
    X = X_train.copy()
    trainy=y_train.copy()
    #X = xx.copy()
    #(trainXX, trainyy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    #X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    #X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]
 
# select real samples
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], {'validity': y, 'label': labels}

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=4):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples) #check these labels!
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    return [images, labels_input], {'validity': y, 'label': labels_input}
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, nmn_label], nmn_y = generate_fake_samples(g_model, latent_dim, n_samples) #TODO!:Numan (nmns were _ and _) - change labels in this row and debug!
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :], cmap='gray_r')
        np.savetxt('test_raw_nc%d%d.csv' % (i,step), X[i,:], delimiter=',')
        np.savetxt('test_cat_nc%d%d.csv' % (i,step), nmn_label[i],delimiter=',')
    # save plot to file
    #np.savetxt('test_raw_nc%d.csv' % (step), X[:,:,0], delimiter=',')
    #np.savetxt('test_cat_nc%d.csv' % (step), nmn_label[:],delimiter=',')
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=30, n_batch=64):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    
    for i in range(n_steps):
        # Train discriminator on real samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        d_loss_real = d_model.train_on_batch(X_real, [y_real, labels_real])

        # Train discriminator on fake samples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss_fake = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        
        # Train generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        #y_gan = {'validity': ones((n_batch, 1)), 'label': z_labels}
        #g_loss = gan_model.train_on_batch([z_input, z_labels], y_gan)
        y_gan = ones((n_batch, 1))
        g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        
        # Summarize loss
        print(f'>Iteration {i+1}/{n_steps}, \n D_real: {d_loss_real:.3f}, \n D_fake: {d_loss_fake:.3f}, \n G: {g_loss[0]:.3f}')
        
        # Performance evaluation
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, latent_dim)
 
# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
discriminator.trainable = True
generator.trainable = True
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)



