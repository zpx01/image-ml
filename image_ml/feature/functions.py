def trainable_feature_seg(img, labels, sigma_min, sigma_max, n_estimators, n_jobs, max_depth, max_samples):

    import numpy as np
    from skimage import data, segmentation, feature, future
    from sklearn.ensemble import RandomForestClassifier
    from functools import partial

    # Build an array of labels for training the segmentation.
    # Here we use rectangles but visualization libraries such as plotly
    # (and napari?) can be used to draw a mask on the image.
    training_labels = labels
    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            multichannel=True)
    features = features_func(img)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs,
                                max_depth=max_depth, max_samples=max_samples)
    clf = future.fit_segmenter(training_labels, features, clf)
    result = future.predict_segmenter(features, clf)
    return result

def find_local_maxima(img):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage import data, img_as_float

    im = img

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=20)

    # # display results
    # fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(im, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('Original')

    # ax[1].imshow(image_max, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('Maximum filter')

    # ax[2].imshow(im, cmap=plt.cm.gray)
    # ax[2].autoscale(False)
    # ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # ax[2].axis('off')
    # ax[2].set_title('Peak local max')

    return image_max, coordinates

def canny_edge_detection(img, sigma):
    """
    Algorithm for edge detection in
    noisy images using Canny filter.
    """
    from skimage import feature
    import numpy as np
    median = np.median(img)
    edges = feature.canny(median, sigma=sigma)
    return edges

def ridge_detection(img, filter="meijering"):
    from skimage.color import rgb2gray
    from skimage.filters import meijering, sato, frangi, hessian

    img = rgb2gray(img)

    if filter == "meijering":
        meijering_img = meijering(img)
        return meijering_img
    elif filter=="sato":
        sato_img = sato(img)
        return sato_img
    elif filter=="frangi":
        frangi_img = frangi(img)
        return frangi_img
    elif filter=="hessian":
        hessian_img = hessian(img)
        return hessian_img
    else:
        return "Filter not found. Filter must be either 'meijering', 'sato', 'frangi', or 'hessian'."
