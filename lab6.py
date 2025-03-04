from PIL import Image, ImageOps

def apply_prewitt_filter(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.grayscale(image)
    image_data = image.load()

    cols, rows = image.size

    kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    kernel_y = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

    padded_image = (
        [[0] * (cols + 2)] +
        [[0] + [image_data[x, y] for x in range(cols)] + [0] for y in range(rows)] +
        [[0] * (cols + 2)]
    )

    gradient_magnitude = [
        [
            min(255, int((
                sum(
                    padded_image[i + di - 1][j + dj - 1] * kernel_x[di][dj] 
                    for di in range(3) for dj in range(3)
                ) ** 2 +
                sum(
                    padded_image[i + di - 1][j + dj - 1] * kernel_y[di][dj] 
                    for di in range(3) for dj in range(3)
                ) ** 2
            ) ** 0.5)) 
            for j in range(1, cols + 1)
        ]
        for i in range(1, rows + 1)
    ]

    result_image = Image.new("L", (cols, rows))
    result_pixels = result_image.load()

    for y in range(rows):
        for x in range(cols):
            result_pixels[x, y] = gradient_magnitude[y][x]

    return result_image

def apply_canny_edge_detector(image_path):
    image = Image.open(image_path).convert("L")
    image_data = image.load()

    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    width, height = image.size

    padded_image = (
        [[0] * (width + 2)] +
        [[0] + [image_data[x, y] for x in range(width)] + [0] for y in range(height)] +
        [[0] * (width + 2)]
    )

    gradient_magnitude = [
        [
            max(0, min(255, int( (
                sum(
                    sobel_x[dy][dx] * padded_image[y + dy - 1][x + dx - 1]
                    for dy in range(3) for dx in range(3)
                )**2 +
                sum(
                    sobel_y[dy][dx] * padded_image[y + dy - 1][x + dx - 1]
                    for dy in range(3) for dx in range(3)
                )**2
            )**0.5)))
            for x in range(1, width + 1)
        ]
        for y in range(1, height + 1)
    ]

    result_image = Image.new("L", (width, height))
    result_pixels = result_image.load()

    for y in range(height):
        for x in range(width):
            result_pixels[x, y] = gradient_magnitude[y][x]

    return result_image

if __name__ == "__main__":
    image_path = "pg.jpg"

    result = apply_canny_edge_detector(image_path)
    result1 = apply_prewitt_filter(image_path)

    result1.show()
    output_path = "prewitt_result.jpg"
    result1.save(output_path)
    result.show()
    output_path = "canny_result.jpg"
    result.save(output_path)


