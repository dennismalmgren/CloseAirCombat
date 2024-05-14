import hydra
import pkgutil
import io
import zipfile
from .acmi import Acmi
import numpy as np
from astropy.timeseries import LombScargle

@hydra.main(version_base="1.1", config_path="", config_name="main_config")
def main(cfg: "DictConfig"):  # noqa: F821
    data = pkgutil.get_data("tacview_analysis.acmi_files","45th-TAW-Round_1.acmi")
    zip_file = io.BytesIO(data)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Contents of the zip file: {file_list}")
        acmi_filename = file_list[0]
        acmi_data = zip_ref.read(acmi_filename)
        acmi_text = acmi_data.decode('utf-8-sig')
        acmi_io = io.StringIO(acmi_text)
        read_first_line = False
        read_second_line = False
        acmi_rep = Acmi()

        for line in acmi_io:
            if line.strip().endswith('\\'):
                print('Escaped line detected, unsupported')
                raise Exception("Unsupported escaped line detected")
            if not read_first_line:
                acmi_rep.parse_first_line(line.strip())
                read_first_line = True
            elif not read_second_line:
                acmi_rep.parse_second_line(line.strip())
                read_second_line = True
            else:
                acmi_rep.parse_line(line.strip())
        print("First time-frame: " + str(acmi_rep.timeframes[0]))
        print("Last time-frame: " + str(acmi_rep.timeframes[-1]))
        np_times = np.asarray(acmi_rep.timeframes)
        deltas = np_times[1:] - np_times[:-1]
        print("Global sampling statistics: ")
        print(f"Min: " + str(np.min(deltas)) + ", Max: " + str(np.max(deltas)) + ", Mean: " + str(np.mean(deltas)) + ", Std: " + str(np.std(deltas)))
        for obj_id in acmi_rep.objects:
            obj = acmi_rep.objects[obj_id]
            if (obj.type() == ['Air', 'FixedWing']):
                print(obj.name())
                print(obj.color()) 
                print("Added at: " + str(obj.added_at))
                first_timeframe = obj.added_at
                if (obj.removed_at is not None):
                    print("Removed, last time frame: " + str(obj.removed_at))
                    last_timeframe = obj.removed_at
                else: 
                    last_timeframe = acmi_rep.timeframes[-1]
                    print("Not removed, last time frame: " + str(last_timeframe))
                print()
                obj_latitudes = []
                obj_timeframes = []
                obj_longitudes = []
                obj_altitudes = []         
                obj_rolls = []           
                for time in acmi_rep.timeframes:
                    if (time < first_timeframe or time > last_timeframe):
                        continue
                    latitude = obj.latitude(time)
                    longitude = obj.longitude(time)
                    altitude = obj.altitude(time)
                    roll = obj.roll(time)
                    obj_timeframes.append(time)
                    obj_latitudes.append(latitude)
                    obj_longitudes.append(longitude)
                    obj_altitudes.append(altitude)
                    obj_rolls.append(roll)
                    #print(f"Time: {time}, Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}")
                obj_latitudes = np.asarray(obj_latitudes)
                obj_longitudes = np.asarray(obj_longitudes)
                obj_altitudes = np.asarray(obj_altitudes)
                obj_timeframes = np.asarray(obj_timeframes)
                obj_rolls = np.asarray(obj_rolls)
                frequency, power = LombScargle(obj_timeframes, obj_rolls).autopower(maximum_frequency=1.0)
                print("Altitudes - min - max: ", min(obj_rolls), " - ", max(obj_rolls))
                import matplotlib.pyplot as plt
                plt.plot(frequency, power)
                plt.xlabel('Frequency')
                plt.ylabel('Power')
                plt.title('Lomb-Scargle Periodogram for ' + obj.name())
                plt.show()
                delta_times = obj_timeframes[1:] - obj_timeframes[:-1]
                print("Sampling time: Min: ", np.min(delta_times), ", Max: ", np.max(delta_times), ", Mean: ", np.mean(delta_times), ", Std: ", np.std(delta_times))
            #    print(obj.latitude())
            #    print(obj.longitude())
        print('done')


if __name__ == "__main__":
    main() 