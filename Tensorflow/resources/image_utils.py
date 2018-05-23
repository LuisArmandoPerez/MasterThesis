import numpy as np
def circular_shift(image:np.ndarray, w_pix_shift:int, h_pix_shift:int):
    """
    Takes a numpy array and shifts it it periodically along the
    width and the height
    :param image: input image, must be at least a 2D numpy array
    :param w_pix_shift: number of pixels that the image is shifted in the width
    :param h_pix_shift: number of pixels that the image is shifted in the height
    :return: numpy array of the shifted image
    """
    # Shift the image along the width
    shifted_image = np.roll(image, w_pix_shift, axis = 1)
    # Shift the image along the height
    shifted_image = np.roll(shifted_image, h_pix_shift, axis = 0)
    return shifted_image



def create_circular_shift_generator(image:np.ndarray, axis:int, batch_size:int, shuffle:bool):
    max_pix_shift = image.shape[axis]
    assert max_pix_shift%batch_size==0, 'The image size along the axis is not a multiple of the batch_size.'
    def circular_shift_generator():
        # Create the order of pixel shifting
        pixel_shifts = np.arange(max_pix_shift)
        if shuffle:
            np.random.shuffle(pixel_shifts)
        # Number of batches per epoch
        batches = max_pix_shift//batch_size
        for num_batch in range(batches):
            data_batch = np.zeros([batch_size, *image.shape], dtype=image.dtype)
            for num_element in range(batch_size):
                data_batch[num_element] = np.roll(image,
                                                  pixel_shifts[(num_batch*batch_size)+num_element],
                                                  axis=axis)
            yield data_batch
    return circular_shift_generator