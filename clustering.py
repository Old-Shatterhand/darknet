import numpy as np

"""
The goal of this script is to find bounding boxes, that match the boxes in the annotated data best for a given size the 
images are resized to. The technique used for this is k-means clustering with the iou as distance metric.
"""

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    :param box1: first bounding box
    :param box2: second bounding box
    :return: fraction of the overlap of the two boxes over the union of the two boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[1] - box1[3] // 2, box1[2] - box1[4] // 2, box1[1] + box1[3] // 2, box1[2] + box1[
        4] // 2
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[1] - box2[3] // 2, box2[2] - box2[4] // 2, box2[1] + box2[3] // 2, box2[2] + box2[
        4] // 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def resize(image, data, target=416):
    """
    Resize the input image to the target-size. The result will be a quadratic image of shape (target, target)

    :param image: Image to resize
    :param data: bounding boxes in YOLO darknet form to be changed
    :param target: target size of the image. The result will be quadratic in this
    :return: resized and in (target,target)-shaped black background embedded image and accordingly adjusted bounding
        boxes
    """
    shape = image.shape
    factor = shape[0] / target
    old_width = shape[1]

    # resize the image
    image = cv2.resize(image, (int(shape[1] / factor), target), interpolation=cv2.INTER_AREA)

    # embed ir into the black background
    pad = (target - image.shape[1]) // 2
    image = cv2.copyMakeBorder(image, 0, 0, pad + (1 if (2 * pad + image.shape[1]) != target else 0), pad,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # adjust the bounding boxes, don't touch this unless it is really necessary and you know what you do
    data_out = []
    for c, cx, cy, w, h in data:
        cx = ((cx * old_width / factor) + pad) / target
        w = w * old_width / (factor * target)
        data_out.append([c, cx, cy, w, h])

    return image, data_out


def read_images(path, pre=False, post=False):
    """
    Reading an image and its according box width annotations. The file with the annotations should be named as the
    image file and located in the same folder, but end with ".txt". Internally uses read_box.

    :param path: path to the image to read in
    :param pre: flag indicating that this is used for preprocessing and one one image is read and no bounding boxes.
    :param post: flag indicating that images for the post processing are read. This flag should only be set in case one
        has to read multiple images for postprocessing, e.g. to read multiple splits of the same image to merge them.
        Otherwise, it is sufficient to set the pre-flag to read a single image without bounding boxes
    :return: image and according bounding boxes in YOLO darknet format as well as the names of the files
    """

    images, boxes, names = [], [], []
    for file in os.listdir(path):
        # only consider image formats and only those images that also have a annotation file
        if (file.endswith(".png") or file.endswith(".jpg")) and \
                (os.path.exists(os.path.join(path, file[:-4] + ".txt")) or post):
            names.append(file[:-4])
            images.append(cv2.imread(os.path.join(path, file)))
            boxes.append(read_boxes(os.path.join(path, file[:-4] + ".txt")))

    return (images, boxes), names



def iou(box1, box2):
    """
    To compute the iou of the two boxes, I just use the bbox_iou method from the augment.py file. Since the boxes are
    represented using only width and height, we can compute the bounding boxes by mimicking rectangles that start in
    (0, 0).
    :param box1: bounding box number 1 to compare
    :param box2: bounding box number 2 to compare
    :return: return iou of two boxes with the given width and height
    """
    return bbox_iou((0, 0, 0, *box1), (0, 0, 0, *box2))


def prepare_boxes(path, size=416):
    """
    Prepare the boxes, one gets from reading in all the images given in the path-directory.
    :param path: path of the directory with the images
    :param size: size to resize the images to
    :return: list of width and height as floats giving the width and height of the resized boxes from the images
    """
    (images, boxes), _ = read_images(path)

    # get the resized boxes
    r_boxes = [resize(image, box)[1] for image, box in zip(images, boxes)]

    # scale the boxes to the width and height of the image as the boxes are currently percentages of the image width
    output = [(size * box[-2], size * box[-1]) for r_box in r_boxes for box in r_box]
    return output


def assignments(centroids, boxes):
    """
    Assignment step of the k-means algorithm. Here, all datapoints are assigned to its nearest centroid.
    :param centroids: current positions of the centroids based on an old assignment
    :param boxes: boxes to be assigned to the centroids.
    :return: the new assignments as list with entries for each box and the sum of all maximal values as estimation of
        how good the assignment captures the boxes
    """
    # initialize and fill a distance matrix using the IoU as distance measure
    dist = np.zeros((len(centroids), len(boxes)))
    for k_iter in range(len(centroids)):
        for b_iter in range(len(boxes)):
            dist[k_iter, b_iter] = iou(centroids[k_iter], boxes[b_iter])

    return list(dist.argmax(axis=0)), sum(dist.max(axis=0)) / len(boxes)


def center(assigns, boxes, k):
    """
    Compute the new centers from the new assignment.
    :param assigns: current assignments of the boxes to the old centroids
    :param boxes: boxes to use to compute the new centroids
    :param k: number of centroids to compute
    :return: new list of k centroids
    """
    centroids = [[0, [0, 0]] for _ in range(k)]

    # sum up all the values of the boxes in each centroid they are assigned to
    for assign, box in zip(assigns, boxes):
        centroids[assign][0] += 1
        centroids[assign][1][0] += box[0]
        centroids[assign][1][1] += box[1]

    # compute the average width and height for each centroid
    return [(int(w[0] / c), int(w[1] / c)) if c != 0 else (0, 0) for c, w in centroids]


def cluster(boxes, k=5, n_iter=1):
    """
    Main routine of the k-means algorithm
    :param boxes: boxes to cluster
    :param k: number of clusters to build
    :param n_iter: number of iterations to perform to find the best (or at least an estimate of the best clustering)
    """
    np.random.seed(42)
    best_score, best_centroids, best_assign = 0, [], []

    # perform the k-means clustering n_iter times
    for i in range(n_iter):

        # shuffle the boxes to get each time a new random initialization
        np.random.shuffle(boxes)
        centroids = boxes[:k]

        # compute the initial assignment
        assigns, score = assignments(centroids, boxes)
        print("Score:", score, end="")

        # recompute the centroids and assignments
        centroids = center(assigns, boxes, k)
        tmp_assigns, score = assignments(centroids, boxes)

        # while the assignments didn't change, perform the last two steps again and again
        while tmp_assigns != assigns:
            assigns = tmp_assigns

            centroids = center(assigns, boxes, k)
            tmp_assigns, score = assignments(centroids, boxes)

        # do some statistics
        print("\t->", score)
        if score > best_score:
            best_score = score
            best_centroids = centroids
            best_assign = tmp_assigns

    # print the best clustering
    print("================================\nBest:")
    print("Score:", best_score)
    print(best_assign)
    print(best_centroids)


def main():
    """
    Initialize the k-mean clustering
    """
    boxes = prepare_boxes("./originalsize", size=416)
    cluster(boxes, k=6, n_iter=100)


if __name__ == '__main__':
    main()
