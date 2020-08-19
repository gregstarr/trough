import h5py
import numpy as np
import apexpy
import pymap3d as pm
import matplotlib.pyplot as plt

from trough import utils, gps

# year month day hour min sec recno kindat kinst ut1_unix ut2_unix pierce_alt gps_site sat_id gdlatr gdlonr los_tec dlos_tec tec azm elm gdlat glon rec_bias drec_bias

with h5py.File("E:\\los_tec\\los_20181001.001.h5", 'r') as f:
    data = f['Data/Table Layout'][()]

ut = data['ut1_unix']
times = ut.astype(int) * np.timedelta64(1, 's') + np.datetime64("1970-01-01T00:00:00")

conv = apexpy.Apex(date=utils.datetime64_to_datetime(times[0]))

el_mask = data['elm'] >= 50
zero_mask = data['tec'] > 0
time_mask = (times >= np.datetime64("2018-10-01T05:00:00")) * (times <= np.datetime64("2018-10-01T10:00:00"))
sat_mask = data['sat_id'] == 8
good_rx = ["inrl", "iafl", "mors", "ptgv", "iaes", "msbv", "insn", "iaol", "mnow", "ksbu", "iatk", "momr", "moct",
           "molj", "iaia", "wimo", "iawt", "mnrr", "wict", "mnhl", "mowe", "wibo", "wicc", "mntr", "nwcc", "kssn",
           "iark", "ilen", "msfl", "ptgx", "wibh", "ilwa", "mscr", "arml", "wial", "mspn", "neb2", "wios", "ileg",
           "al23", "iamy", "iahp", "wiao", "mnmh", "wibr", "widm", "wicf", "corc", "ksol", "iacd", "iamk", "kslw",
           "iawi", "iaha", "wish", "intn", "iacs", "mncd", "nehu", "arsg", "iahu", "wioc", "wipw", "lchs", "wiml",
           "iamc", "wirn", "iagr", "wimn", "iaao", "wisy", "iast", "wirp", "wipi", "indy", "mopy", "mohr", "iame",
           "mnwd", "p803", "mnki", "ildk", "mnai", "repc", "mnsy", "iaau", "iale", "mnmd", "cvms", "iary", "wibi",
           "wijv", "wiws", "dubo", "picl", "widp", "albe", "migr", "p775", "wica", "mspe", "wisp", "mnba", "mosa",
           "momm", "ksog", "wipd", "wiwa", "iapa", "mair", "ilrd", "wigr", "alfa", "chuc", "mnky", "wiwe", "iawe",
           "moc1", "wimi", "iady", "ilfu", "momn", "wimr", "iaky", "iawl", "iad2", "iane", "mocb", "mohi", "iaho",
           "mnnu", "wirc", "ilmo", "wihv", "mome", "mstu", "iacm", "mnsr", "nmkm", "wimh", "incs", "mobd", "inmh",
           "moky", "mnwi", "mogv", "mocv", "misv", "wiaf", "iaub", "mnsw", "moci", "arct", "mssn", "rlap", "iahm",
           "monn", "ilna", "msbt", "wiar", "mnpv", "jfws", "momd", "talc", "iamw", "hcex", "cjtr", "mnpi", "msns",
           "alnc", "ianv", "alds", "mcty", "ranc", "iasg", "mnrs", "moby", "ilmc", "arcy", "iapy", "iaar", "p777",
           "iaon", "iawy", "mnsa", "iact", "inls", "mony", "kstk", "hdil", "ilsh", "mnst", "mnbp", "ilev", "ilwk",
           "iapj", "iacx", "mnae", "hces", "iaay", "wibu", "moad", "iaka", "ialw", "kshw", "mnro", "wisu", "mnho",
           "moki", "chur", "mnlr", "mnrw", "chu2", "okow", "mshs", "wiag", "iaig", "ksbr", "mnm2", "mnpy", "mnwo",
           "wiwr", "msox", "wimf", "wine", "gjoc", "wihi", "bake"]
rx_mask = np.in1d(data['gps_site'], good_rx)
mask = el_mask * zero_mask * time_mask * rx_mask * sat_mask
data = data[mask]
times = times[mask]

el = data['elm']
az = data['azm']

plt.figure()
prof_ax = plt.axes()
plt.figure()
map_ax = plt.axes()
# for rx in ['dubo', 'mnm2', 'nehu', 'mnmh', 'ksog', 'okow']:
mlats = []
grads = []
for rx in good_rx:
    m = data['gps_site'] == rx.encode()
    if m.sum() < 10:
        print(rx, m.sum())
        continue
    print(rx, m.sum())
    rxloc = np.array((data[m]['gdlatr'][0], data[m]['gdlonr'][0], 0))[None, :]
    # plt.plot(rxloc[0, 1], rxloc[0, 0], '.')
    # plt.text(rxloc[0, 1], rxloc[0, 0], rx)
    rxecef = np.column_stack(pm.geodetic2ecef(rxloc[:, 0], rxloc[:, 1], rxloc[:, 2]))[0]
    satecef = np.column_stack(pm.aer2ecef(az[m], el[m], 1000, rxloc[:, 0], rxloc[:, 1], rxloc[:, 2]))
    ipp = gps.get_pierce_points(satecef, rxecef, 350)
    lla = pm.ecef2geodetic(ipp[:, 0], ipp[:, 1], ipp[:, 2])
    mlat, mlon = conv.convert(lla[0], lla[1], 'geo', 'apex')
    g = np.gradient(data[m]['tec'], mlat)
    prof_ax.plot(mlat, g, '.')
    map_ax.scatter(mlon, mlat, s=5, c=data[m]['tec'], vmin=0, vmax=6)
    mlats.append(mlat)
    grads.append(g)

m_vals = np.arange(40, 80, .5)
mlat = np.concatenate(mlats)
grad = np.concatenate(grads)
r = np.ones((m_vals.shape[0], 2)) * np.nan
for i, m in enumerate(m_vals):
    r[i, 0] = m
    mask = (mlat > m - .5) * (mlat <= m + .5)
    if mask.sum() < 1:
        continue
    r[i, 1] = np.median(grad[mask])
plt.figure()
plt.plot(r[:, 0], r[:, 1])
plt.show()
