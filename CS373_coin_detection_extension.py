# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

def RGBPixelArraysToGreyscaleArray(image_width, image_height, px_array_r, px_array_g, px_array_b):

    greyImage = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            greyImage[i][j] = round(0.3 * px_array_r[i][j] + 0.6 * px_array_g[i][j] + 0.1 * px_array_b[i][j])

    return greyImage

def filter(pixel_array, image_width, image_height, kernel, kernel_sum):

    filtered = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            sum = 0.0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    sum += pixel_array[i + a][j + b] * kernel[a + 1][b + 1]
            filtered[i][j] = sum/kernel_sum
    
    return filtered

def computeLaplacianFilter(pixel_array, image_width, image_height):

    laplace = [
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]

    kernel_sum = 1

    edge_image = filter(pixel_array, image_width, image_height, laplace, kernel_sum)

    for i in range(image_height):
        for j in range(image_width):
            edge_image[i][j] = abs(edge_image[i][j])

    return edge_image

def computeMeanBlurring(pixel_array, image_width, image_height):
    
    smoothed_image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)

    for i in range(2, image_height-2):
        for j in range(2, image_width-2):
            total = 0
            for a in range(-2, 3):
                for b in range(-2, 3):
                    total += pixel_array[i + a][j + b]
            
            smoothed_image[i][j] = abs(total / 25)
            
    return smoothed_image

def computeThreshold(pixel_array, threshold_value, image_width, image_height):

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] >= threshold_value:
                pixel_array[i][j] = 255
            else:
                pixel_array[i][j] = 0
    return pixel_array

def computeDilation(pixel_array, image_width, image_height):
    
    dilated_image = createInitializedGreyscalePixelArray(image_width, image_height)

    kernel = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]
        
    for i in range(image_height):
        for j in range(image_width):
            
            for a in range(-2, 3):
                for b in range(-2, 3):
                    
                    x = i + a
                    y = j + b
                    
                    if 0 <= x < image_height and 0 <= y < image_width and kernel[a+2][b+2] == 1 and pixel_array[x][y] == 255:
                        dilated_image[i][j] = 255
                        break
                else:
                    continue
                
                break

    return dilated_image

def computeErosion(pixel_array, image_width, image_height):
    
    eroded_image = createInitializedGreyscalePixelArray(image_width, image_height)

    kernel = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]
        
    for i in range(image_height):
        for j in range(image_width):
            
            neighbors = []
            
            for a in range(-2, 3):
                for b in range(-2, 3):
                    
                    x = i + a
                    y = j + b
                    
                    if 0 <= x < image_height and 0 <= y < image_width and kernel[a+2][b+2] == 1:
                        neighbors.append(pixel_array[x][y])
 
            if 0 not in neighbors:
                eroded_image[i][j] = 255
            
    return eroded_image

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    def get_neighbors(x, y):
        neighbors = []
        if y > 0: neighbors.append((x, y - 1))                 # Left
        if y < image_width - 1: neighbors.append((x, y + 1))   # Right
        if x > 0: neighbors.append((x - 1, y))                 # Up
        if x < image_height - 1: neighbors.append((x + 1, y))  # Down
        return neighbors

    label_image = createInitializedGreyscalePixelArray(image_width, image_height)
    label_count = {}
    label_id = 0
    visited = set()

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] in (255, 1) and (i, j) not in visited:
                label_id += 1
                queue = Queue()
                queue.enqueue((i, j))
                visited.add((i, j))
                label_image[i][j] = label_id
                pixel_count = 0

                while not queue.isEmpty():
                    x, y = queue.dequeue()
                    pixel_count += 1

                    for nx, ny in get_neighbors(x, y):
                        if (nx, ny) not in visited and pixel_array[nx][ny] in (255, 1):
                            queue.enqueue((nx, ny))
                            visited.add((nx, ny))
                            label_image[nx][ny] = label_id

                if pixel_count >= 10000:
                    label_count[label_id] = pixel_count
                else:
                    for x in range(image_height):
                        for y in range(image_width):
                            if label_image[x][y] == label_id:
                                label_image[x][y] = 0

    return label_image, label_count

def computeBoundingBoxBoundaries(label_image, label_count, image_width, image_height):
    bounding_box_list = []
    coin_type_list = []

    min_x = {label: image_width for label in label_count}
    min_y = {label: image_height for label in label_count}
    max_x = {label: 0 for label in label_count}
    max_y = {label: 0 for label in label_count}

    for i in range(image_height):
        for j in range(image_width):
            label = label_image[i][j]
            if label > 0:
                if j < min_x[label]:
                    min_x[label] = j
                if j > max_x[label]:
                    max_x[label] = j
                if i < min_y[label]:
                    min_y[label] = i
                if i > max_y[label]:
                    max_y[label] = i

    for label in label_count:
        width = max_x[label] - min_x[label] + 1
        height = max_y[label] - min_y[label] + 1
        bounding_box_radius = (width + height) / 4

        approximate_circle_area = bounding_box_radius * bounding_box_radius * math.pi
        tolerance = approximate_circle_area * 0.1

        if abs(label_count[label] - approximate_circle_area) <= tolerance:
            bounding_box = [min_x[label], min_y[label], max_x[label], max_y[label]]
            bounding_box_list.append(bounding_box)
            if label_count[label] < 39000:
                coin_type_list.append('10 Cent')
            elif label_count[label] < 41000:
                coin_type_list.append('20 Cent')
            elif label_count[label] < 45000:
                coin_type_list.append('1 Dollar')
            elif label_count[label] < 54000:
                coin_type_list.append('50 Cent')
            elif label_count[label] < 100000:
                coin_type_list.append('2 Dollar')
            else:
                coin_type_list.append('Unknown')


    return bounding_box_list, coin_type_list

def main(input_path, output_path):
    image_name = 'easy_case_6'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    greyImage = RGBPixelArraysToGreyscaleArray(image_width, image_height, px_array_r, px_array_g, px_array_b)

    edgesImage = computeLaplacianFilter(greyImage, image_width, image_height)
    
    blurredImage = edgesImage
    for _ in range(2):
        blurredImage = computeMeanBlurring(blurredImage, image_width, image_height)

    thresholdImage = computeThreshold(blurredImage, 20, image_width, image_height)

    bounding_box_list = []

    erodedImage = thresholdImage
    for _ in range(4):
        erodedImage = computeErosion(erodedImage, image_width, image_height)

    dilatedImage = erodedImage
    for _ in range(4):
        dilatedImage = computeDilation(dilatedImage, image_width, image_height)

    labelImage, label_count = computeConnectedComponentLabeling(dilatedImage, image_width, image_height)
    
    bounding_box_list, coin_type_list = computeBoundingBoxBoundaries(labelImage, label_count, image_width, image_height)

    px_array = greyImage
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')

    if len(bounding_box_list) != 0:
        for i, bounding_box in enumerate(bounding_box_list):
            bbox_min_x = bounding_box[0]
            bbox_min_y = bounding_box[1]
            bbox_max_x = bounding_box[2]
            bbox_max_y = bounding_box[3]

            bbox_xy = (bbox_min_x, bbox_min_y)
            bbox_width = bbox_max_x - bbox_min_x
            bbox_height = bbox_max_y - bbox_min_y
            rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
            axs.add_patch(rect)
        
            # Display the coin type on the top of the bounding box
            coin_label = coin_type_list[i]
            axs.text(bbox_min_x, bbox_min_y, coin_label, color='yellow', fontsize=12, 
                     verticalalignment='bottom', horizontalalignment='left')

    # Adding number of coins detected on the top right corner
    num_coins = len(bounding_box_list)
    axs.text(0.95, 0.95, f'Coins: {num_coins}', transform=axs.transAxes, color='yellow', fontsize=20, 
             verticalalignment='top', horizontalalignment='right')

    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
    
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
