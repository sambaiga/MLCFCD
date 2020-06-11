import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def visualize(events_appliance_id, a, on_event, names, l):
    ev = np.zeros(len(a['current']))
    ev[events_appliance_id[0]] = 0.8
    ev[events_appliance_id[1]] = 2
    ev[events_appliance_id[2]] = 0.95
    ev[events_appliance_id[3]] = 0.95
    if len(on_event)>4:
        ev[events_appliance_id[4]] = 2
    if len(on_event)>5:
        ev[events_appliance_id[5]] = 1
    if len(on_event)>6:
        ev[events_appliance_id[6]] = 2
    plt.plot(a['current'])
    plt.plot(ev)
    plt.text(events_appliance_id[0],1,l[0])
    plt.text(events_appliance_id[1],2.5,l[1])
    plt.text(events_appliance_id[2],1.5,l[2])
    plt.text(events_appliance_id[3],0.85,l[3])
    if len(on_event)>4:
        plt.text(events_appliance_id[4],2,l[4])
    if len(on_event)>5:
        plt.text(events_appliance_id[5],1,l[5])

    if len(on_event)>6:
        #plt.text(400000,2,names[l[4]-1])
        plt.text(events_appliance_id[6],4,l[6])
    plt.show()
    
    
def get_multilabel(on_event, l):
    labels = []
    for event_num in range(len(on_event)):
        if on_event[event_num]==1:
            if event_num==0:
                label=[l[event_num]]
                labels.append(label)
                
            if event_num==1:
                label=[l[event_num-1], l[event_num]]
                labels.append(label)

            if event_num==2 and len(on_event)>4:
                label=[l[event_num-2], l[event_num-1], l[event_num]]
                labels.append(label)

            if event_num==3 and len(on_event)==7:
                label=[l[event_num-3], l[event_num-2], l[event_num-1], l[event_num]]
                labels.append(label)
            #print(f'{label}:{event_num}')


        if on_event[event_num]==0:
            event_id = event_num-len(l)//2
            if event_num==3 and len(on_event)==4:
                label=[l[-1]]
                labels.append(label)
                #print(f'{label}:{event_num}')
                
            if event_num==3 and len(on_event)==5:
                label=[l[-1]]
                labels.append(label)
                #print(f'{label}:{event_num}')

            if event_num==3 and len(on_event)>5:
                label=[l[event_id+1], l[event_id+2]]
                labels.append(label)
                #print(f'{label}:{event_num}')

            if event_num==4 and len(on_event)==6:
                label=[l[i] for i in range(2, 3)]
                labels.append(label)
                #label=[names[l[1]-1, l[2]-1, l[3]-1, l[4]-1]]
                #print(f'{label}:{event_num}')

            if event_num==4 and len(on_event)==7:
                label=[l[-1], l[-2]]
                labels.append(label)
                #label=[names[l[1]-1, l[2]-1, l[3]-1, l[4]-1]]
                #print(f'{label}:{event_num}')

            if event_num==5 and len(on_event)==7:
                label=[l[-1]]
                labels.append(label)
                #label=[names[l[1]-1, l[2]-1, l[3]-1, l[4]-1]]
                #print(f'{label}:{event_num}')
                
    return labels

def read_events_labels(file1, file2, data_path='/home/ibcn079/data/Data/'):
    a = pd.read_csv(data_path+file1,header=None)
    a = a.values

    b = pd.read_csv(data_path+file2,header=None)
    b = b.values

    events = {}
    labels = {}
    for index in range(len(a)):
        line = a[index]
        ev = [int(i) for i in line[0].strip().split(" ")]
        events[index] = ev

        line = b[index]
        ev = [int(i) for i in line[0].strip().split(" ")]
        labels[index] = ev
    return events, labels  
        
def load_multilabel_plaid(data_path='/home/ibcn079/data/Data/'):
    folder = "FINALAGGREGATED"
    file1 = folder + '_events'
    file2 = folder + '_labels'

    final_currents = []
    final_voltages = []
    i_max = []
    targets = []
    count = 0
    events, labels = read_events_labels(file1, file2, data_path)
    with tqdm(total=len(events)) as pbar:
        for index, e, l in zip(events.keys(), events.values(), labels.values()):
            events_appliance_id = np.array(e)

            #print(count)

            f = data_path+folder+'/'+str(index+1)
            a = pd.read_csv(f,names=['current','voltage'])

            if 9 in l:
                on_event = [1,1,1,0,0] if len(e)==5 else [1,1,1,1,0,0,0]
            else:
                on_event =[1,1,0,0] if len(e)==4 else [1,1,1,0, 0,0]
            #multi_labels = create_multi_labels(appliances, on_pattern, on_events)
            #visualize(events_appliance_id, a, on_event, names, l)
            labels=get_multilabel(on_event, l)
            targets +=labels
            #print(labels)
            #print(l)
            #input("Enter")
            #display.clear_output()

            for idx in range(len(events_appliance_id)):
                if idx!=len(events_appliance_id)-1:
                    I =a['current'].values[events_appliance_id[idx]:events_appliance_id[idx+1]]
                    U =a['voltage'].values[events_appliance_id[idx]:events_appliance_id[idx+1]]
                    s = int(len(I)*0.5)
                    e = int(len(I)*0.75)

                    final_currents += [I[s:e][:500]]
                    final_voltages += [U[s:e][:500]]
                    i_max += [I[s:e].max(0)]

                    """
                    plt.subplot(1,2,1)
                    plt.plot(U, I)
                    plt.subplot(1,2,2)
                    plt.plot(U[s:e], I[s:e])
                    #plt.axvline(events_appliance_id[idx]+s, color="y")
                    plt.show()
                    input("Enter")
                    display.clear_output()
                    """
            
            pbar.set_description('processed: %d' % (1 + count))
            pbar.update(1)
            count+=1
        pbar.close()
        
    print(f"currents size:{len(final_currents)}")  
    print(f"labels size:{len(targets)}")
    print(f"voltage:{len(final_voltages)}") 
    print(f"max_current:{len(i_max)}")
    
    assert len(final_currents)==len(final_voltages)==len(targets)

    return np.array(final_currents), np.array(final_voltages), np.array(targets), np.array(i_max)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    folder_id = "plaid"
    path = "/home/ibcn079/data/Data/"
    
    c, v, y, i_max=load_multilabel_plaid(data_path=path)
    np.save(f"data/{folder_id}/current.npy", c)
    np.save(f"data/{folder_id}/voltage.npy", v)
    np.save(f"data/{folder_id}/labels.npy", y)
    np.save(f"data/{folder_id}/max_current.npy", i_max)