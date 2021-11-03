def time_sine_cosine(t):
    T_freq = [10]
    n_days = 1
    while (T_freq[-1] < 60*24*7*n_days):
        T_freq.append(T_freq[-1] * 2)
    omegas = [2*np.pi/t_freq for t_freq in T_freq]
    enc = []
    for om in omegas:
        enc = enc + [np.sin(om*t), np.cos(om*t)]
    return enc

def time_external(t):
    in_t = t
    mins = in_t % 60
    in_t = in_t // 60
    hrs = in_t % 24
    in_t = in_t // 24
    days = in_t % 7
    in_t = in_t // 7
    weeks = in_t
    return [weeks, days, hrs, mins]

def time_external_normalized(t):
    in_t = t
    mins = in_t % 60
    in_t = in_t // 60
    hrs = in_t % 24
    in_t = in_t // 24
    days = in_t % 7
    in_t = in_t // 7
    weeks = in_t
    return [weeks, days/7, hrs/24, mins/60]