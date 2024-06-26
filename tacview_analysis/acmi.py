import sys
import os
import zipfile
import codecs
import datetime
import sortedcontainers


class Object:
    def __init__(self, id, added_at):
        self.id = id
        self.removed_at = None
        self.added_at = added_at
        self.data = {}
        self.sampling_timeframes = sortedcontainers.SortedSet()

    def set_value(self, field, timeframe, val):
        if field not in self.data:
            self.data[field] = sortedcontainers.SortedDict()
        self.data[field][timeframe] = val
        self.sampling_timeframes.add(timeframe)

    def value(self, field: str, time=None):
        if field not in self.data:
            return None

        if time is not None:
            key_ind =  self.data[field].bisect_left(time)
            if key_ind >= len(self.data[field]):
                key_ind = -1
        else:
            key_ind = -1
        key = self.data[field].keys()[key_ind]
        return self.data[field][key]

    def group(self, time=None):
        return self.value("Group", time)

    def x(self, time=None):
        return self.value("x", time)

    def y(self, time=None):
        return self.value("y", time)

    def longitude(self, time=None):
        return self.value("Longitude", time)

    def latitude(self, time=None):
        return self.value("Latitude", time)

    def altitude(self, time=None):
        return self.value("Altitude", time)
    
    def roll(self, time=None):
        return self.value("roll", time)
    
    def type(self, time=None):
        return self.value("Type", time)

    def name(self, time=None):
        return self.value("Name", time)
    
    def color(self, time=None):
        return self.value("Color", time)
    
    def coalition(self, time=None):
        return self.value("Coalition", time)
    
    def __str__(self):
        return "{id}: '{name}' {long}, {lat}, {alt}".format(
            id=self.id,
            name=self.value("Name"),
            long=self.value("Longitude"),
            lat=self.value("Latitude"),
            alt=self.value("Altitude"))


class Frame:
    def __init__(self, time):
        self.time = time
        self.objects = {}


class Acmi:

    def __init__(self):
        self.file_version = None
        self.file_type = None

        # global properties
        self.data_source = None
        self.data_recorder = None
        self.reference_time = None
        self.recording_time = None
        self.author = None
        self.title = None
        self.category = None
        self.briefing = None
        self.debriefing = None
        self.comments = None
        self.reference_longitude = None
        self.reference_latitude = None
        self.authentication_key = None

        self.objects = {}
        self.timeframes = []
        self.cur_reftime = 0.0

    @staticmethod
    def split_fields(line):
        fields = []
        i = 1
        lastfield = 0
        while i < len(line):
            if line[i-1] != '\\' and line[i] == ',':
                fields.append(line[lastfield:i])
                lastfield = i + 1
            i += 1

        fields.append(line[lastfield:i])
        return fields

    def _parse_global_property(self, fields):
        for field in fields[1:]:  # skip objid (0)
            (prop, val) = field.split('=', 1)
            if prop == "ReferenceTime":
                self.reference_time = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%SZ")
            elif prop == "RecordingTime":
                self.recording_time = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%fZ")
            elif prop == "ReferenceLongitude":
                self.reference_longitude = float(val)
            elif prop == "ReferenceLatitude":
                self.reference_latitude = float(val)
            elif prop == "DataSource":
                self.data_source = val
            elif prop == "DataRecorder":
                self.data_recorder = val
            elif prop == "Author":
                self.author = val
            elif prop == "Title":
                self.title = val
            elif prop == "Category":
                self.category = val
            elif prop == "Briefing":
                self.briefing = val
            elif prop == "Debriefing":
                self.debriefing = val
            elif prop == "Comments":
                self.comments = val
            elif prop == "AuthenticationKey":
                self.authentication_key = val
            else:
                raise RuntimeError("Unknown global property: " + prop)

    def _update_object(self, obj_id: int, timeframe: float, fields):
        if obj_id not in self.objects:
            self.objects[obj_id] = Object(obj_id, timeframe)

        obj = self.objects[obj_id]
        for field in fields[1:]:
            (prop, val) = field.split('=', 1)
            if prop == "T":
                pos = val.split('|')
                if pos:
                    if len(pos) == 3:
                        if pos[0]:
                            obj.set_value("Longitude", timeframe, self.reference_longitude + float(pos[0]))
                        if pos[1]:
                            obj.set_value("Latitude", timeframe, self.reference_latitude + float(pos[1]))
                        if pos[2]:
                            obj.set_value("Altitude", timeframe, float(pos[2]))
                    elif len(pos) == 5:
                        if pos[0]:
                            obj.set_value("Longitude", timeframe, self.reference_longitude + float(pos[0]))
                        if pos[1]:
                            obj.set_value("Latitude", timeframe, self.reference_latitude + float(pos[1]))
                        if pos[2]:
                            obj.set_value("Altitude", timeframe, float(pos[2]))
                        if pos[3]:
                            obj.set_value("x", timeframe, float(pos[3])) #u
                        if pos[4]:
                            obj.set_value("y", timeframe, float(pos[4])) #v
                    elif len(pos) == 6:
                        if pos[0]:
                            obj.set_value("Longitude", timeframe, self.reference_longitude + float(pos[0]))
                        if pos[1]:
                            obj.set_value("Latitude", timeframe, self.reference_latitude + float(pos[1]))
                        if pos[2]:
                            obj.set_value("Altitude", timeframe, float(pos[2]))
                        if pos[3]:
                            obj.set_value("roll", timeframe, float(pos[3]))
                        if pos[4]:
                            obj.set_value("pitch", timeframe, float(pos[4]))
                        if pos[5]:
                            obj.set_value("yaw", timeframe, float(pos[5]))
                    elif len(pos) == 9:
                        if pos[0]:
                            obj.set_value("Longitude", timeframe, self.reference_longitude + float(pos[0]))
                        if pos[1]:
                            obj.set_value("Latitude", timeframe, self.reference_latitude + float(pos[1]))
                        if pos[2]:
                            obj.set_value("Altitude", timeframe, float(pos[2]))
                        if pos[3]:
                            obj.set_value("roll", timeframe, float(pos[3]))
                        if pos[4]:
                            obj.set_value("pitch", timeframe, float(pos[4]))
                        if pos[5]:
                            obj.set_value("yaw", timeframe, float(pos[5]))
                        if pos[6]:
                            obj.set_value("x", timeframe, float(pos[6]))
                        if pos[7]:
                            obj.set_value("y", timeframe, float(pos[7]))
                        if pos[8]:
                            obj.set_value("heading", timeframe, float(pos[8]))
                    else:
                        raise Exception('Invalid transformation format detected')
            elif prop == "Name":
                obj.set_value(prop, timeframe, val)
            elif prop == "Parent" or prop == "FocusTarget" or prop == "LockedTarget":
                obj.set_value(prop, timeframe, int(val, 16))
            elif prop == "Type":
                obj.set_value(prop, timeframe, val.split("+"))
            elif prop in ["Pilot", "Group", "Country", "Coalition",
                          "Color", "Registration", "Squawk", "Debug", "Label"]:
                obj.set_value(prop, timeframe, val)
            # numeric except coordinates start here
            # floats
            elif prop in ["Importance", "Length", "Width", "Height",
                          "IAS", "CAS", "TAS", "Mach", "AOA", "HDG",
                          "HDM", "Throttle", "RadarAzimuth", "RadarElevation",
                          "RadarRange", "LockedTargetAzimuth",
                          "LockedTargetElevation", "LockedTargetRange", "Flaps", "LandingGear",
                          "AirBrakes"]:
                obj.set_value(prop, timeframe, float(val))
            # int
            elif prop in ["Slot", "Afterburner", "Tailhook",
                          "Parachute", "DragChute", "RadarMode",
                          "LockedTargetMode"]:
                obj.set_value(prop, timeframe, int(val))
            else:
                print("Unknown property:", prop)

        
    def parse_first_line(self, rawline):
        if rawline.startswith('FileType='):
            self.file_type = rawline[len('FileType='):].strip()
        else:
            raise RuntimeError("ACMI file doesn't start with FileType.")
        
    def parse_second_line(self, rawline):
        if rawline.startswith('FileVersion='):
            self.file_version = float(rawline[len('FileVersion='):].strip())
            if self.file_version < 2.1:
                raise RuntimeError("Unsupported file version: {v}".format(v=self.file_version))
        else:
            raise RuntimeError("ACMI file missing FileVersion.")
    
    def parse_line(self, rawline):
        line = rawline.strip()  # type: str
        if not line or line.startswith('//'):
            return  # ignore comments
        if line.startswith('#'):
            self.cur_reftime = float(line[1:])
            self.timeframes.append(self.cur_reftime)
            return
        if line.startswith('-'):
            obj_id = int(line[1:], 16)
            self.objects[obj_id].removed_at = self.cur_reftime
            return
        fields = self.split_fields(line)
        obj_id = int(fields[0], 16)
        if obj_id == 0:
            self._parse_global_property(fields)
        else:
            self._update_object(obj_id, self.cur_reftime, fields)
    
    # def _parse(self, fp):
    #     with fp as f:
    #         ar = AcmiFileReader(f)
    #         rawline = next(ar)
    #         if rawline.startswith('FileType='):
    #             self.file_type = rawline[len('FileType='):].strip()
    #         else:
    #             raise RuntimeError("ACMI file doesn't start with FileType.")

    #         rawline = next(ar)
    #         if rawline.startswith('FileVersion='):
    #             self.file_version = float(rawline[len('FileVersion='):].strip())
    #             if self.file_version < 2.1:
    #                 raise RuntimeError("Unsupported file version: {v}".format(v=self.file_version))
    #         else:
    #             raise RuntimeError("ACMI file missing FileVersion.")

    #         cur_reftime = 0.0
    #         linenr = 2
    #         for rawline in ar:
    #             linenr += 1
    #             line = rawline.strip()  # type: str
    #             if not line or line.startswith('//'):
    #                 continue  # ignore comments

    #             if line.startswith('#'):
    #                 cur_reftime = float(line[1:])
    #                 self.timeframes.append(cur_reftime)
    #                 continue

    #             if line.startswith('-'):
    #                 obj_id = int(line[1:], 16)
    #                 self.objects[obj_id].removed_at = cur_reftime
    #             else:
    #                 fields = self.split_fields(line)
    #                 obj_id = int(fields[0], 16)

    #                 #print(obj_id, fields)
    #                 if obj_id == 0:
    #                     self._parse_global_property(fields)
    #                 else:
    #                     self._update_object(obj_id, cur_reftime, fields)

    def object_ids(self):
        return self.objects.keys()

    def alive_objects(self):
        return [self.objects[objkey] for objkey in self.objects if self.objects[objkey].removed_at is None]

    def removed_objects(self):
        return [self.objects[objkey] for objkey in self.objects if self.objects[objkey].removed_at is not None]

    def __str__(self):
        return str(
            {
                "FileType": self.file_type,
                "FileVersion": self.file_version,
                "DataSource": self.data_source,
                "DataRecorder": self.data_recorder,
                "ReferenceTime": self.reference_time.isoformat(),
                "RecordingTime": self.recording_time.isoformat(),
                "Author": self.author,
                "Title": self.title,
                "Category": self.category,
                "Briefing": self.briefing,
                "Debriefing": self.debriefing,
                "Comments": self.comments,
                "ReferenceLongitude": self.reference_longitude,
                "ReferenceLatitude": self.reference_latitude
            }
        )


if __name__ == "__main__":
    acmi = Acmi()
    acmi.load(sys.argv[1])

    #import pyproj
    #dcs_proj = pyproj.Proj("+proj=tmerc +lat_0=0 +lon_0=33 +k_0=0.9996 +x_0=-99517 +y_0=-4998115")

    print(acmi.object_ids())
    print(acmi.timeframes)
    for o in acmi.alive_objects():
        if "Air" in o.type():
            print(o)
            print(o.x(), o.y())

    print(acmi)