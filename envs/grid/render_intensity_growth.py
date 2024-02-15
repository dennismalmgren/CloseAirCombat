import matplotlib.pyplot as plt
import numpy as np

def test_slicing_native():
    num_envs = 2
    sensor_width = 5
    sensor_halfwidth = sensor_width // 2
    sensor_range = 7
    agent_loc = np.asarray([[5, 4], 
                      [12, 8]],
                      dtype=np.int32)
    dir = np.asarray([0, 1], dtype=np.int32)
    #loc_h = 5
    #loc_w = 5
    area_width = 40
    area_height = 20


    sensor_area_h = np.asarray([
        [- sensor_range, 0],
        [ - sensor_halfwidth, sensor_halfwidth + 1],
        [1, sensor_range + 1],
        [ - sensor_halfwidth, sensor_halfwidth + 1]
    ], dtype=np.int32)

    sensor_area_w = np.asarray([
        [ - sensor_halfwidth, sensor_halfwidth + 1],
        [1, sensor_range + 1],
        [ - sensor_halfwidth, sensor_halfwidth + 1],
        [ - sensor_range, 0]
    ], dtype=np.int32)

    sensor_area_coverage_h = np.expand_dims(agent_loc[..., 0], -1) + sensor_area_h[dir]
    sensor_area_coverage_w = np.expand_dims(agent_loc[..., 1], -1) + sensor_area_w[dir]
    h_indices = np.expand_dims(np.expand_dims(np.arange(area_height), 0), -1)
    w_indices = np.expand_dims(np.expand_dims(np.arange(area_width), 0), 0)
 
    mask_h = (h_indices >= np.expand_dims(sensor_area_coverage_h[..., 0:1], -1)) & (h_indices <= np.expand_dims(sensor_area_coverage_h[..., 1:2], -1))
    mask_w = (w_indices >= np.expand_dims(sensor_area_coverage_w[..., 0:1], -1)) & (w_indices <= np.expand_dims(sensor_area_coverage_w[..., 1:2], -1))
    sensor_coverage_mask = mask_h & mask_w

    print('ok')

def test_slicing():
    sensor_width = 5
    sensor_halfwidth = sensor_width // 2
    sensor_range = 7
    loc = np.asarray([[5, 4], 
                      [12, 8]],
                      dtype=np.int32)
    dir = np.asarray([0, 1], dtype=np.int32)
    #loc_h = 5
    #loc_w = 5
    area_width = 40
    area_height = 20
    max_offset = max(sensor_range, sensor_halfwidth) 
    task_area_coverage = np.ones((area_height, area_width), dtype=np.bool_)
    padded_task_area = np.zeros((area_height + 2 * max_offset, area_width + 2 * max_offset), dtype=np.bool_)
    #add padding.
    #I assume it's always range, though.
    #new area is old area + 2x max offset in width and height.
    #so a (0, 0) coordinate in old area will be at (max_offset, max_offset)
    #this means any sensor coverage is still at 0 or better.
    loc_offset = loc + max_offset
    sensor_area_h = np.asarray([
        [- sensor_range, 0],
        [ - sensor_halfwidth, sensor_halfwidth + 1],
        [1, sensor_range + 1],
        [ - sensor_halfwidth, sensor_halfwidth + 1]
    ], dtype=np.int32)

    sensor_area_w = np.asarray([
        [ - sensor_halfwidth, sensor_halfwidth + 1],
        [1, sensor_range + 1],
        [ - sensor_halfwidth, sensor_halfwidth + 1],
        [ - sensor_range, 0]
    ], dtype=np.int32)

    sensor_area_coverage_h = np.expand_dims(loc_offset[..., 0], -1) + sensor_area_h[dir]
    sensor_area_coverage_w = np.expand_dims(loc_offset[..., 1], -1) + sensor_area_w[dir]
    h_indices = np.expand_dims(np.expand_dims(np.arange(area_height), 0), -1)
    w_indices = np.expand_dims(np.expand_dims(np.arange(area_width), 0), 0)
 
    mask_h = (h_indices >= np.expand_dims(sensor_area_coverage_h[..., 0:1], -1)) & (h_indices <= np.expand_dims(sensor_area_coverage_h[..., 1:2], -1))
    mask_w = (w_indices >= np.expand_dims(sensor_area_coverage_w[..., 0:1], -1)) & (w_indices <= np.expand_dims(sensor_area_coverage_w[..., 1:2], -1))
    mask = mask_h & mask_w
    padded_task_area[mask] = 1 #we can skip this step.
#    padded_task_area[sensor_area_coverage_h[0]:sensor_area_coverage_h[1], sensor_area_coverage_w[0]:sensor_area_coverage_w[1]] = 1
    task_area_coverage &= padded_task_area[max_offset:area_height + max_offset, max_offset:area_width + max_offset] 

    print('ok')

def main():
    #birth rate
    br = 0.1
    #survivability rate
    ps = 0.99
    #total time
    T = 1000
    expected_limit = br / (1 - ps)
    print("Expected limit:", expected_limit)
    ninety_percent = expected_limit * 0.95
    rise_time = -1
    time_to_one = -1
    arrivals = [0]
    time = [0]
    for t in range(1, T):
        next_value = arrivals[-1] * ps + br
        if next_value >= 1 and time_to_one == -1:
            time_to_one = t
        if next_value > ninety_percent and rise_time == -1:
            rise_time = t
        arrivals.append(arrivals[-1] * ps + br)
        time.append(t)

    print("Rise time: ", rise_time)
    print("Time to one: ", time_to_one)
    plt.plot(time, arrivals)
    if rise_time != -1:
        plt.axvline(rise_time, color = 'r', linestyle = '--', label = f"95% rise time ({rise_time})")
    if time_to_one != -1:
        plt.axvline(time_to_one, color = 'b', linestyle = '--', label = f"Time to 1 expected arrival: ({time_to_one})")
    plt.xlabel("Time")
    plt.ylabel("Expected Arrivals")
    plt.legend()
    plt.show()

if __name__ == "__main__":
   # test_slicing_native()
    main()
