import pandas as pd


def FsmMetParser(file, freq="1h", resample=False):
    '''
    Parses FSM forcing files from toposcale sims
    '''

    df = pd.read_csv(file, delim_whitespace=True,
                     header=None, index_col='datetime',
                     parse_dates={'datetime': [0, 1, 2, 3]},
                     date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d %H'))

    df.columns = ['ISWR', 'ILWR', 'Sf', 'Rf', 'TA', 'RH', 'VW', 'P']

    if resample == "TRUE":
        df = df.resample(freq).apply(resample_func)

    return(df)


def FsmSnowParser(file, freq="1H", resample=False):
    '''
    parses FSM output fuiles
    '''

    df = pd.read_csv(file, delim_whitespace=True, parse_dates=[[0, 1, 2]], header=None)
    df.set_index(df.iloc[:, 0], inplace=True)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.columns = ['albedo', 'Rof', 'HS', 'SWE', 'TS10', 'TS50']

    if resample == "TRUE":
        df = df.resample(freq).apply(resample_func)

    return(df)

def FsmPlot(df):
    df.plot(subplots=True)
    plt.show()

def SmetParser(file, resample=True, freq='1H', NA_val=-999):
    '''
    val_file = full path to a smet
    resample = "TRUE" or "FALSE"
    freq = '1D' '1H' etc

    '''

    # find statrt of data
    lookup = '[DATA]'
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                skip = num

    df = pd.read_csv(file, delim_whitespace=True, index_col='datetime',
                     parse_dates={'datetime': [0]}, header=None, skiprows=skip, na_values=NA_val)

    # get col names
    lookup = 'fields'
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                fields = num

    import linecache
    myfields = linecache.getline(file, fields)
    colnames = myfields.split("= ")[1]
    cnames = colnames.split(" ")

    df.columns = cnames[1:]

    if resample == "TRUE":
        df = df.resample(freq).apply(resample_func)

    return(df)


def point2sample(lat, lon):
    """
    Function to find which sample a point exists in.

    Args:
        lon: longitude of point
        lat: latitude of point

    Returns:
        sampleN: sample number
    """

def validate_meteo(smetfile:
    wfj = SmetParser("/home/joel/data/wfj_optimal/WFJ_optimaldataset_v8.smet")

    # find sample closest to validation station

