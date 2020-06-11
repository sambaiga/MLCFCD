import numpy as np

def select_zc(voltage, Ts):
    zero_crossing = np.where(np.diff(np.sign(voltage)))[0]
    
    if voltage[zero_crossing[0]+1] > 0:
        zero_crossing = zero_crossing[0:]
    else:
        zero_crossing = zero_crossing[1:]
        
    if len(zero_crossing) % 2 == 1:
        zero_crossing = zero_crossing[:-1]
        
    if zero_crossing[-1] + Ts >= len(voltage):
        zero_crossing = zero_crossing[:-2]
        
    return zero_crossing


def transform(current, voltage, on_event, Ts):
    

    zc = select_zc(voltage, Ts)

    before_event = np.concatenate([current[zc[0]:zc[0]+Ts//2],current[zc[1]:zc[1]+Ts//2]])
    after_event = np.concatenate([current[zc[-2]:zc[-2]+Ts//2],current[zc[-1]:zc[-1]+Ts//2]])

    if on_event:
        c = after_event - before_event
        v = np.concatenate([voltage[zc[-2]:zc[-2]+Ts//2],voltage[zc[-1]:zc[-1]+Ts//2]])
        
    else:
        c = before_event - after_event
        v = np.concatenate([voltage[zc[0]:zc[0]+Ts//2],voltage[zc[1]:zc[1]+Ts//2]])
        
    return c, v
