from .load_data import *
from .visualizer_functions import *
from .transform_functions import *
from .functions import *
from .processing_functions import *
import random
from tqdm import tqdm


def get_correct_labels_lilac(labels):
    correct_1_phase_motor = [920,923,956, 959, 961, 962, 1188]
    correct_hair = [922, 921, 957, 958,  960, 963, 1181, 1314]
    correct_bulb = [1316]
    
    correct_labels = []
    for idx, l in enumerate(labels):
        if idx in correct_1_phase_motor:
            correct_labels.append('1-phase-async-motor')
        elif idx in correct_hair:
            correct_labels.append('Hair-dryer')
        elif idx in correct_bulb:
            correct_labels.append('Bulb')
        else:
            correct_labels.append(l)
    correct_labels = np.hstack(correct_labels)
    return correct_labels


classes = list(appliance_names)
events = {"1": ([10, 15, 20], [40, 45, 50]),
          "2": ([10, 40], [160, 190]),
          "3": ([10, 15], [45, 50]),
          "4": ([10, 20], [50, 60]),
          "5": ([10, 40], [220, 250]),
          "6": ([10, 20], [80, 90]),
          "7": ([10, 20], [50, 80]),
          "8": ([10, 110], [190, 220]),
          "9": ([10, 40], [190, 220]),
          "10": ([10, 20], [50, 60]),
          "11": ([10, 15], [75, 80]),
          "12":  ([10, 15], [40, 45])
          }
def get_data(full_path, file_id):
    
    file_name = os.path.basename(full_path)
    names = file_name.strip().split(".")[0].strip().split("_")
    appliances = names[:-1]

    on_pattern = names[-1]
    on_pattern = [int(x) for x in on_pattern]
    if len(on_pattern) == 3:
        on_pattern = on_pattern * 2

    readings = read_tdms(full_path)
    df = pd.DataFrame(columns=['I1', 'V1', 'I2', 'V2', 'I3', 'V3']+appliances)

    for k in range(1, 4):

        df['I{}'.format(k)] = readings['I{}'.format(k)]
        df['V{}'.format(k)] = readings['V{}'.format(k)]

    t = np.arange(len(df))*2
    on_time = [x*1e5 for x in events[file_id][0]]
    off_time = [x*1e5 for x in events[file_id][1]]

    slices = []
    apps = []
    windows_on = []
    windows_off = []
    for i in range(len(appliances)):
        slices.append(slice(np.where(t == on_time[i])[
            0][0], np.where(t == off_time[i])[0][0]))
        apps.append(np.zeros_like(t))
        windows_on.append(np.where(t == on_time[i])[0][0])
        windows_off.append(np.where(t == off_time[i])[0][0])

    for i in range(len(appliances)):
        apps[i][slices[i]] = 1

    on_events = np.zeros_like(t)

    for window in windows_on:
        on_events[window] = 1

    for window in windows_off:
        on_events[window] = -1

    df["on-events"] = on_events
    
    events_ids = df[(df["on-events"] == 1) | (df["on-events"] == -1)]["on-events"]
    df = df[["I1", "I2", "I3", "V1", "V2", "V3"]]
    
    voltages = ["V1", "V2", "V3"]
    currents = ["I1", "I2", "I3"]
    c = df[currents].values
    v = df[voltages].values
    
    #events = events.index.values
    lst = [df]
    del df
    del lst
    df =pd.DataFrame()

    return c, v, events_ids, appliances,  on_pattern

def to_categorical(num_classes=len(classes)):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[[i for i in range(num_classes)]]


def multilabel_hot_encoding(appliance, num_classes=len(classes)):
    labels=to_categorical(num_classes)
    app_label=np.zeros(num_classes).astype(np.int8)
    for app in appliance:
        id=classes.index(app)
        app_label+=labels[id]
    return app_label

def multilabel_hot_decoding(encoding):
    index, = np.where(encoding == 1)
    appliance=np.array(classes)[index]
    return appliance

def get_multilabels(apps, on_pattern, event_num, on_event):
    appliances = appliances_check(apps)
    
    if on_event==1:
        if event_num==0:
            appliance=[appliances[on_pattern[event_num]-1]]

        if event_num==1:
            appliance=[appliances[on_pattern[event_num-1]-1], appliances[on_pattern[event_num]-1]]
            
        if event_num==2 and len(appliances)==3:
            appliance=[appliances[on_pattern[event_num-2]-1], appliances[on_pattern[event_num-1]-1], appliances[on_pattern[event_num]-1]]


        
    else:
        if event_num==0 and len(appliances)==2:
            appliance=[appliances[on_pattern[event_num]-1]]
            
        if event_num==0 and len(appliances)==3:
            appliance=[appliances[on_pattern[event_num+1]-1], appliances[on_pattern[event_num+2]-1]]
            
        if event_num==1 and len(appliances)==2:
            appliance=None
            
        if event_num==1 and len(appliances)==3:
            appliance=[appliances[on_pattern[event_num+1]-1]]
            
        if event_num==2:
            appliance=None
            


    return appliance

def create_multi_labels(appliances, on_pattern, on_events):
    
    labels = []
    for event_num in range(len(on_events)):
   
        if event_num>=len(appliances):
            event_id = event_num-len(appliances)
            event_type=-1
        else:
            event_id=event_num
            event_type=1



        appliance=get_multilabels(appliances, on_pattern, event_id, on_events[event_num])

        if appliance is not None:

            #encoding=multilabel_hot_encoding(appliance)
            labels.append(appliance)
            #decoding=multilabel_hot_decoding(encoding)
            #print(f"{decoding}: {encoding}")
    return labels
    #print(event_id)
    
def get_activations(full_path, file_id,  trans=None,verbose=False, vis=True):
    period = int(50e3/50)
    cycles = period*10
    
    current_appliance_id = []
    voltage_appliance_id = []
    power_appliance_id = []
    
    current_signal = []
    
    
    

    final_currents = []
    final_voltages = []
    final_power = []
    i_max = []
    targets = []

    c, v, events, appliances,  on_pattern = get_data(full_path, file_id)
    events_appliance_id = events.index.values
    on_events = np.where(events.values == 1, 1, 0).tolist()
    #apps_name = [appliances[on_pattern[i]-1] for i in range(len(on_pattern))]
    #apps_name = appliances_check(apps_name)
    appliances = appliances_check(appliances)
    #power = get_three_phase_power(df)
    

    multi_labels = create_multi_labels(appliances, on_pattern, on_events)
    #labels   = apps_name
    #power_appliance_id+=[power_ids]
   
    
    for idx in range(len(on_events)):
       
        
        
        if idx!=len(on_events)-1:
            #print(multilabel_hot_decoding(multi_labels[idx]))
            #targets +=[multi_labels[idx]]
            i =c[events_appliance_id[idx]:events_appliance_id[idx+1]]
            u =v[events_appliance_id[idx]:events_appliance_id[idx+1]]
            s = int(len(i)*50/100)
            e = int(len(u)*75/100)
            i = i[s:e][:1000]
            u = u[s:e][:1000]
            
            """"
            if idx==0:
                power=p[power_ids[idx]-cycles:power_ids[idx]]
                current_signal=c[events_appliance_id[idx]-cycles:events_appliance_id[idx]]
            else:
                power=p[power_ids[idx-1]:power_ids[idx]]
                current_signal=c[events_appliance_id[idx-1]:events_appliance_id[idx]]
            
            final_power+=[power]
            """
           
            if trans is None:
                i_max += [i.max(0)]
                final_currents += [i]
                final_voltages += [u]
                
                if vis:
                    plt.plot(u[:1000, 0], i[:1000, 0])
                    plt.show()
                
            elif trans:
                i_ct = isc_transform(i)
                v_ct = isc_transform(u)
                i_max += [i_ct.max(0)]
                final_currents += [i_ct[:1000]]
                final_voltages += [v_ct[:1000]]
                #i_max += [i.max(0)]
                
                if vis:
                    plt.plot(i_ct[:1000, 2])
                    plt.show()
                
        
    return final_currents, final_voltages, multi_labels, i_max


def select_aggregate_data_appliance_type(path, trans=2, verbose=False, vis=False):
    
    current = []
    current_signal =[]
    voltage = []
    power = []
    labels = []
    states = []
    max_current = []
    power_events = []

    files = getListOfFiles(path)
    print(len(files))

    print("Load data")
    files_id = 0
    with tqdm(total=len(files)) as pbar:
        for root, k, fnames in sorted(os.walk(path)):
            file_id = root.strip().split("/")[-1]
            if file_id:
                for j, fname in enumerate(sorted(fnames)):
                    full_path = os.path.join(root, fname)
                    #print(fname)
                    c, v, l, c_max = get_activations(full_path, file_id, trans, verbose, vis)
                    current += c
                    voltage += v
                    labels += l
                    max_current += c_max
                
                    pbar.set_description('processed: %d' % (1 + files_id))
                    pbar.update(1)
                    files_id += 1
        pbar.close()
        

    print(f"currents size:{len(current)}")  
    print(f"labels size:{len(labels)}")
    print(f"voltage:{len(voltage)}") 
    print(f"max_current:{len(max_current)}")
    
    assert len(current)==len(voltage)==len(labels)
         
    return np.array(current), np.array(voltage), labels,  np.array(max_current)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    folder_id = "lilac"
    path = f"/home/ibcn079/data/LILAC/Triple/"
    path = "../data/Triple/"
    
    c, v, y, i_max=select_aggregate_data_appliance_type(path,   trans=None,  verbose=False, vis=False)
    np.save(f"data/{folder_id}/current.npy", c)
    np.save(f"data/{folder_id}/voltage.npy", v)
    np.save(f"data/{folder_id}/max_currents.npy", i_max)
    np.save(f"data/{folder_id}/labels.npy", np.array(y))
   
    
