import numpy as np


def yield_samples(n,
                  add_blob=None,
                  alpha=0.5,
                  image_size=32,
                  noise_mean=1,
                  noise_std=0.1,
                  blob_size=16,
                  blob_fill_value=0,
                  blob_mean=0,
                  blob_delta=1):
    """
    Generate images containing an alternating pattern (blob).


    Arguments:
    n -- size of batch to yield

    Keyword arguments:
    add_blob -- True: all images with blob; False: all without; None: 50/50
    alpha -- weight; overall factor for blob when added to noise
    image_size -- width/height in pixels (images are square); needs to be even
    noise_mean -- mean of the normally distributed noise term
    noise_std -- standard deviation of the normally distributed noise term
    blob_size -- width/height of the blob (blobs are square); needs to be even
    blob_fill_value -- fill value for pixels outside the blob
    blob_mean -- baseline value for the blob
    blob_delta -- values in the blob will alternate by delta around mean

    Returns:
    Generator yielding batches of the form X, y
    """

    blob_fill = blob_fill_value * np.ones((n, image_size, image_size))

    blob_alt = np.kron(np.ones((int(blob_size/2), int(blob_size/2))),
                       np.array([[1, -1], [-1, 1]]))
    blob_alt = blob_delta * blob_alt

    blob_mea = blob_mean * np.ones((blob_size, blob_size))

    while True:
        noise = np.random.normal(noise_mean, noise_std,
                                 (n, image_size, image_size))

        ind_x = np.random.randint(0, image_size-blob_size+1, size=(n,))
        ind_y = np.random.randint(0, image_size-blob_size+1, size=(n,))

        blob_base = blob_fill
        blob_osc = np.zeros((n, image_size, image_size))

        for i in range(n):
            blob_base[i, ind_x[i]:ind_x[i]+blob_size,
                      ind_y[i]:ind_y[i]+blob_size] = blob_mea
            blob_osc[i, ind_x[i]:ind_x[i]+blob_size,
                     ind_y[i]:ind_y[i]+blob_size] = blob_alt

        if add_blob is None:
            factors = np.random.choice(np.array([0, -1, +1], dtype=np.int8),
                                       p=[0.5, 0.25, 0.25], size=(n, 1, 1))
        elif add_blob:
            factors = np.random.choice(np.array([-1, +1], dtype=np.int8),
                                       p=[0.5, 0.5], size=(n, 1, 1))
        elif not add_blob:
            factors = np.zeros((n, 1, 1), dtype=np.int8)

        y = (factors != 0).astype(factors.dtype)
        X = noise + alpha*(np.multiply(factors, blob_osc)
                           + np.multiply(y, blob_base))
        X = np.expand_dims(X, -1)
        y = y.reshape((-1,1))
        yield X, y
