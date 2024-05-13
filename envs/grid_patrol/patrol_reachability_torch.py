import torch

def manhattan(start_point, end_grid_h, end_grid_w):
    return torch.abs(start_point[0] - end_grid_h) + torch.abs(start_point[1] - end_grid_w)
    
def fill_in_dir(distances, index_grid_h, index_grid_w, height, width, start_dir):
    for start_h in range(height):
        for start_w in range(width):
            start_point = torch.tensor([start_h, start_w])

            #ending to the top
            end_dir = start_dir
            #MH region (top)
            end_h = slice(0, start_h)
            end_w = slice(0, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH + 2 region (left)
            end_h = slice(start_h, height - 1)
            end_w = slice(0, max(0, start_w - 1))
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 2 region (right)
            end_h = slice(start_h, height - 1)
            end_w = slice(start_w + 2, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 4 region
            end_h = slice(start_h, height - 1)
            end_w = slice(max(0, start_w - 1), start_w + 2)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 4
            #Starting point
            distances[start_h, start_w, start_dir, start_h, start_w, end_dir] = 0
            #bottom row
            end_h = slice(height - 1, height)
            end_w = slice(0, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = torch.inf

            #ending to the right
            end_dir = (start_dir + 1) % 4
            #MH region (top right)
            end_h = slice(0, start_h + 1)
            end_w = slice(start_w + 1, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH region (bottom right)
            end_h = slice(start_h + 1, height)
            end_w = slice(start_w + 2, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH + 2 region (top left)
            end_h = slice(0, start_h)
            end_w = slice(1, start_w + 2)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 2 region (bottom left)
            end_h = slice(start_h + 1, height)
            end_w = slice(1, start_w + 2)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 4 region (left)
            end_h = slice(start_h, start_h + 1)
            end_w = slice(1, start_w)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 4
            #Starting point
            distances[start_h, start_w, start_dir, start_h, start_w, end_dir] = 4
            #left column
            end_h = slice(0, height)
            end_w = slice(0, 1)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = torch.inf

            #ending to the bottom
            end_dir = (start_dir + 2) % 4
            #MH region (bottom left)
            end_h = slice(start_h + 1, height)
            end_w = slice(0, start_w)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH region (bottom right)
            end_h = slice(start_h + 1, height)
            end_w = slice(start_w + 1, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH + 2 region (top left)
            end_h = slice(0, start_h + 1)
            end_w = slice(0, start_w)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 2 region (top right)
            end_h = slice(0, start_h + 1)
            end_w = slice(start_w + 1, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 4 region (top)
            end_h = slice(0, start_h)
            end_w = slice(start_w, start_w + 1)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 4
            #MH + 4 region (bottom)
            end_h = slice(start_h + 1, height)
            end_w = slice(start_w, start_w + 1)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 4
            #Starting point
            distances[start_h, start_w, start_dir, start_h, start_w, end_dir] = 4
            #top row
            end_h = slice(0, 1)
            end_w = slice(0, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = torch.inf

            #ending to the left
            end_dir = (start_dir + 3) % 4
            #MH region (top left)
            end_h = slice(0, start_h + 1)
            end_w = slice(0, start_w)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH region (bottom left)
            end_h = slice(start_h + 1, height)
            end_w = slice(0, max(0, start_w - 1))
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w])
            #MH + 2 region (top right)
            end_h = slice(0, start_h)
            end_w = slice(start_w, width - 1)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2
            #MH + 2 region (bottom right)
            end_h = slice(start_h + 1, height)
            end_w = slice(max(0, start_w - 1), width - 1)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 2            #MH + 4 region (left)
            #MH + 4 region (right)
            end_h = slice(start_h, start_h + 1)
            end_w = slice(start_w + 1, width - 1)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = manhattan(start_point, index_grid_h[end_h, end_w], index_grid_w[end_h, end_w]) + 4
            #Starting point
            distances[start_h, start_w, start_dir, start_h, start_w, end_dir] = 4
            #right column
            end_h = slice(0, height)
            end_w = slice(width - 1, width)
            distances[start_h, start_w, start_dir, end_h, end_w, end_dir] = torch.inf

def calc_distance_matrix(start_height, start_width):
    dirs = 4
    distances = torch.zeros((start_height, start_width, dirs, start_height, start_width, dirs))

    for start_dir in range(4):
        if start_dir == 0:
            height = start_height
            width = start_width
        elif start_dir == 1:
            height = start_width
            width = start_height
            distances = torch.rot90(distances, k = 1, dims=[0, 1])
            distances = torch.rot90(distances, k = 1, dims=[3, 4])
        elif start_dir == 2:
            height = start_height
            width = start_width
            distances = torch.flip(distances, dims=[0])
            distances = torch.flip(distances, dims=[3])
        elif start_dir == 3:
            height = start_width
            width = start_height
            distances = torch.rot90(distances, k = 3, dims=[0, 1])
            distances = torch.rot90(distances, k = 3, dims=[3, 4])
        
        heights = torch.arange(height)
        widths = torch.arange(width)
        index_grid = torch.meshgrid(heights, widths, indexing='ij')
        index_grid_h = index_grid[0]
        index_grid_w = index_grid[1]
        fill_in_dir(distances, index_grid_h, index_grid_w, height, width, start_dir)

        if start_dir == 1:
            distances = torch.rot90(distances, k = 1, dims=[1, 0])
            distances = torch.rot90(distances, k = 1, dims=[4, 3])
        elif start_dir == 2:
            distances = torch.flip(distances, dims=[0])
            distances = torch.flip(distances, dims=[3])
        elif start_dir == 3:
            distances = torch.rot90(distances, k = 3, dims=[1, 0]) 
            distances = torch.rot90(distances, k = 3, dims=[4, 3])
    return distances