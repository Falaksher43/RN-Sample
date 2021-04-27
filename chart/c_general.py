import chart
import utils_df
import utils_db
import saccades
import librosa
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matviz.matviz as mv
from matviz.matviz import etl
from matviz.matviz.helpers_graphing import *
from PIL import Image
from collections import OrderedDict
from collections import namedtuple
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
from scipy import interpolate, stats
import matplotlib.ticker as mtick

import traceback

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)

C = linspecer(3)

# todo: create the sdf in prosaccade analysis and export, instead of creating in charts
#  = analysis.p_results_to_sdf(processed_data)
import analysis

def plot_data_quality(df):
    # current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    # processed_data = current_df.iloc[0]['processed']

    subplotter(2, 3, [0, 1, 2])
    title('blink detection, good data selection')
    blink_detection(df, legender='none')

    subplotter(2, 3, 3)
    plot_instructions(df)

    # plot the sampling rate
    subplotter(2, 3, 4)
    plot_ds_hist(df)

    subplotter(2, 3, 5)
    plot_ds_cdf(df, legender='none')


def process_visit(complete_df):
    plot_funcs = chart.get_exam_vizs()
    for index, df in complete_df.iterrows():
        pfuncs = plot_funcs[df.exam.lower()]
        for func in pfuncs:
            try:
                plt.figure(figsize=(8,5))
                func(complete_df, df.visit_exam_id)
                plt.close()
                print('Successfully plotted: {},{} - {}'.format(df.exam, func, df.visit_exam_id))
            except:
                print('Issue plotting: {},{} - {}'.format(df.exam, func, df.visit_exam_id))


def plot_eye_speed(z, t='none', c=[0, 0, 0], frac_c=0.2, w_fac=30, nfac=5):
    """
    Plot the eye movements along with charts for their speed.
    https://stackoverflow.com/questions/19390895/matplotlib-plot-with-variable-line-width

    :param z: the eye positions
    :param t: (optional) times associated with those positions
    :param c: main color
    :param frac_c: contrast factor for main color converted to background color
    :param w_fac: width fraction for the eye speeds conversion to widths
    :param nfac: how many times you want to interpolate the wide lines
    :return:
    """
    if len(c) != 3:
        raise Exception("plot_eye_speed does not accept time, only z")

    c2 = np.array(c) * frac_c + (1 - frac_c)
    v = np.abs(np.diff(z))
    to_plot = 1 + v * w_fac

    if type(t) == str:
        nfac = 1

    if nfac == 1:
        points = np.array([np.real(z), np.imag(z)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=to_plot, color=c2)
        gca().add_collection(lc)
    else:
        t_idx = np.arange(len(to_plot))
        # super sampling factor
        t_idx_extra = np.arange(0, max(t_idx), 1.0 / nfac)
        t_extra = np.interp(t_idx_extra, t_idx, t[:-1])

        # interpolate the widths with pchip (and 0 for nans)
        # so missing data is interpreted as 0 velocity and has no trail
        to_plot[np.isnan(to_plot)] = 0
        f = interpolate.PchipInterpolator(t[:-1], to_plot)
        to_plot_i = f(t_extra)

        # interpolate the positions linearly
        xi = np.interp(t_extra, t, np.real(z))
        yi = np.interp(t_extra, t, np.imag(z))
        zi = xi + 1j * yi

        # make the chart
        points = np.array([np.real(zi), np.imag(zi)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=to_plot_i, color=c2)
        gca().add_collection(lc)

    cplot(z, '-', c=c)
    axis('equal')

    return gca()


def nhist_error_radial(t, z, rad_ylim):
    radial = np.abs(z)
    _ = ndhist(t + 1j * radial, maxy=rad_ylim, miny=0, normx=True, fx=4, levels=[50, 95], smooth=1)


def plot_error_radial(t, z, stim, rad_ylim):
    plot(t, np.abs(z), color=linspecer(1)[0])
    plot(t, np.abs(stim), 'k--', lw=2)

    ylim([0, rad_ylim])
    xlabel('Time (s)')
    ylabel('radial distance')

    legend(['eye', 'stim'], loc='lower right')
    nicefy()


def plot_error_angular(t, z, stim):
    y = np.rad2deg(np.angle(z) - np.angle(stim))
    I = y > 180
    y[I] = y[I] - 360
    I = y <= -180
    y[I] = y[I] + 360
    plot(t, y, '.', color=linspecer(1)[0])
    # plot(t, np.rad2deg(np.angle(stim)), 'k--', lw=2)
    mv.viz.plot_zero(linecolor='k')

    xlabel('Time (s)')
    ylabel('angular error (deg)')

    legend(['eye', 'stim'], loc='lower right')
    nicefy()


def nhist_error_angular(t, z, stim):
    angular = np.rad2deg(np.angle(z) - np.angle(stim))
    _ = ndhist(t + 1j * angular, normx=True, fx=4, miny=-90, maxy=90, levels=[50, 95], smooth=1)


def create_single_bar(y, yerr, x_pos=0):
    gca().bar(x_pos, y, yerr=yerr,
              align='center', alpha=0.1, ecolor='black', capsize=5, lw=2,
              width=0.5, error_kw={'lw': 2, 'capthick': 2, 'alpha': .1}, color='gray')


def plot_exam_history(processed_df):
    user_df = utils_db.get_complete_df_for_user(processed_df.iloc[0]['subject_id'])

    unique_dates = user_df.groupby(pd.Grouper(key='created_date', freq='D')).apply(len)
    unique_years = np.unique(user_df.created_date.dt.year)
    daylabels = ['M', '', 'W', '', 'F', '', 'S']

    fig_sizer(4, 18)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=len(unique_years), figure=fig)

    for i in range(len(unique_years)):
        subplot = fig.add_subplot(spec[i, 0:2])
        calmap.yearplot(unique_dates, ax=subplot, year=unique_years[i], vmin=0, vmax=8, daylabels=daylabels)
        subplot.set_ylabel(unique_years[i])

    nicefy()

    return 'png'


def plot_ds(df):
    fig_sizer(8, 20)

    # plot the sampling rate
    subplotter(1, 2, 0)
    plot_ds_hist(df)

    # plot the line charts to see that data recorded is changing
    subplotter(1, 2, 1)
    plot_ds_cdf(df)


def plot_ds_hist(df):
    nhist(df['Delta Time'], normalize='frac', std_times=.1, f=2, noerror=True)
    xlabel('Delta time (s)')
    title("Sampling rate histogram")
    format_axis_date()
    nicefy()


def plot_ds_cdf(df, legender=None):
    C = linspecer(len(utils_df.KEY_COLUMNS))
    x = 100 * np.arange(len(df) - 1) / len(df)
    for ii, cur_col in enumerate(utils_df.KEY_COLUMNS):
        y = np.sort(np.abs(np.diff(df[cur_col])))
        plot(x, y, '.-', c=C[ii])

    if legender is None:
        legend(utils_df.KEY_COLUMNS, loc='best')


    xlabel('rank')
    ylabel('difference between sucessive values')

    title('sampling rate inspection')
    set_axis_ticks_pctn()
    yscale('log')
    xlabel('rank')
    ylabel('difference between sucessive values')
    format_axis_date()
    nicefy()


def blink_detection(df, legender=None):
    def plot_nans(t, w, height=3, *args, **kargs):
        I = np.isnan(w)
        plot(t[I], height * np.ones(sum(I)), '^r', markersize=15, markeredgewidth=0, *args, **kargs)

    t = df['Total Time']
    pupil = df['Pupil Diameter']
    Lx = df['Left Eye Direction X']
    Rx = df['Right Eye Direction X']
    Ly = df['Left Eye Direction Y']
    Ry = df['Right Eye Direction Y']

    LRx = df['Combine Eye Direction X']
    LRy = df['Combine Eye Direction Y']

    fig_sizer(8, 13)

    C = linspecer(5)
    C2 = brighten(C, .7)

    tr_do = (3, 0, 2 * 90)
    tr_up = (3, 0, 0 * 90)
    sq = (4, 0, 45)
    sq2 = (4, 0, 0)

    plot_nans(t, Lx, height=+.03, marker=tr_up, c=C2[0])
    plot_nans(t, LRx, height=0, marker=sq, c=C2[0])
    plot_nans(t, Rx, height=-.03, marker=tr_do, c=C2[0])

    plot_nans(t, Ly, height=-.3 + .03, marker=tr_up, c=C2[1], label='_nolegend_')
    plot_nans(t, LRy, height=-.3, marker=sq, c=C2[1])
    plot_nans(t, Ry, height=-.3 - .03, marker=tr_do, c=C2[1], label='_nolegend_')

    plot(t, Lx, c=C[0], zorder=10, label='_nolegend_')
    plot(t, -Rx, c=C[0], zorder=10, label='_nolegend_')

    plot(t, Ly - .3, c=C[1], zorder=10, label='_nolegend_')
    plot(t, -Ry - .3, c=C[1], zorder=10, label='_nolegend_')

    # include pupil data yay!
    plot(t, pupil / 10 - .2, c=C[4], label='_nolegend_')
    plot_nans(t, pupil, height=.3, c=C[4], marker=sq2)

    # mabye we can logical_or reduce on the DF directly???
    master_nan = np.logical_or.reduce(np.isnan([Lx, LRx, Rx, Ly, LRy, Ry, pupil]))
    # Expand the region of nans by 4 on each side!
    master_nan = nan_smooth(master_nan, np.array(9 * [1]) / 9).astype(bool)

    nicefy(touch_limits=True)

    events = etl.start_and_ends(master_nan)
    plot_range_idx(t, events, color=brighten(3 * [.5]))

    if legender is None:
        legend(['Eye direction X Left',
                'Eye direction X Combined',
                'Eye direction X Right',
                'Eye direction Y Combined', 'Pupil'], loc='center right')

    yticks([])
    xlabel('time')

    print(ylim())


def plot_xy(df, std_times=2):
    # ds = np.median(df['Delta Time'])
    x = df['Left Eye Position X'].values
    y = df['Left Eye Position Y'].values

    I = (x != 0) & (y != 0)
    x = x[I]
    y = y[I]

    plot(x, y, '.')

    xlim(std_times * np.nanstd(x) * np.array([-1, 1]) + np.nanmean(x))
    ylim(std_times * np.nanstd(y) * np.array([-1, 1]) + np.nanmean(y))
    # print(lags[np.argmax(corrs)] * 1000)


def plot_instructions(df):
    """
    Plot the period of the exam when the instructions dot appears
    :param df:
    :return:
    """
    if 'Instruction Dot Size' in df:
        x, y, t, stim = utils_df.get_inst_xy_t_stim(df)
        # start_idx, start_time = hlp.get_exam_start(df)
        # stim = df['Instruction Dot Size'].values[:start_idx]
        # stim_t = df['Total Time'].values[:start_idx]

        plot(t, x - x[-1], '.-')
        plot(t, y - y[-1], '.-')
        # plot(t, stim / 110000.)
        plot(t, stim / 240.)
        # plot(t, stim)
        nicefy(expand_y=True)
        ylabel('X/Y position')
        xlabel('time (s)')
        legend(['eye x', 'eye y', 'dot size'], loc="best")
        # legend(['eye x', 'eye y', 'dot size'], loc="lower right")
        title("Instructions following")
        nicefy(expand_y=True)
    else:
        clf()
        title("Instruction Dot Data Missing")


def save_all_plots(df, cur_folder):
    os.mkdir('figs')
    plot_xy(df)
    print_fig_fancy(os.path.join(cur_folder,'figs','plot_xy.jpg'))


def categorize_population_by_visit_exam_id(x, cindex, pindex):
    if x.visit_exam_id in pindex.values:
        return 'control'
    elif x.visit_exam_id == cindex:
        return 'subject'
    else:
        return np.nan


def custom_cmap(cvals, colors):
    """
    Creates a custom color map.

    Parameters:
    cvals (array): Array of values indicating where color should change.
    colors (list): List of colors (e.g. ['red', 'green', 'red]) representing
                    colors to change at corresponding index of cvals.

    Returns:
    cmap: matplotlib cmap object
    """
    norm = plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', tuples)
    return cmap


def view_colormap(cvals, cmap, size='half'):
    """Plot a colormap"""
    norm = plt.Normalize(min(cvals),max(cvals))
    colors = cmap(np.linspace(norm(np.min(cvals)), norm(np.max(cvals))))

    if size == 'full':
        plt.subplots(figsize=(8,1.25), tight_layout=True)
    else:
        plt.subplots(figsize=(4,1.25), tight_layout=True)
    plt.imshow([colors], extent=[np.min(cvals), np.max(cvals), 0, .15], aspect='auto')
    plt.xlim([np.min(cvals), np.max(cvals)])
    plt.yticks([])


def plot_blinks(complete_df, user_id, subj_legend='subject'):
    #   separate population and the subject
    blink_df = complete_df[['user_id', 'blinks', 'exam']]
    df_pop, df_subj = [w for _, w in blink_df.groupby(blink_df['user_id'] == user_id)]

    if len(df_subj) > 1:
        logger.warning("found " + len(df_subj) + " subject rows, using the first")

    short_id = user_id[-5:]

    fig_sizer(5, 10)
    blinks = df_pop['blinks'].values
    n = len(blinks)
    rank = 100 * np.arange(n) / n
    y = sorted(blinks)
    plot(y, rank, 'k', label='_nolegend_')
    xlim([0, 50])
    gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    xlabel('number of blinks')
    ylabel('percent of exams')

    plot(2 * [df_subj.iloc[0]['blinks']], [90, 0], '-v', markersize=14, c='r', markevery=2)
    title(df_subj.iloc[0]['exam'] + ', ' + short_id)
    legend([subj_legend + " " + short_id], loc='lower right')


def plot_saccades(df, **kargs):
    t, z, stim = utils_df.get_t_z_stim(df)

    saccades_df = saccades.cget_saccades(z, t, **kargs)

    # plot the line charts to see that data recorded is changing
    subplotter(4, 1, [0, 1, 2])

    plot_eye_speed(z, t, c=[0.5, .5, 0.9])

    for idx, row in saccades_df.iterrows():
        idx = list(row['saccades'])
        cplot(z[idx], ':k', alpha=.5, lw=4)
        cplot(z[idx],'k')
        cplot(z[idx[1]], 'ok', fillstyle='none', markersize=10)
        cplot(z[idx[1]], 'ok', markersize=10, alpha=.3)

    axis('equal')
    nicefy()

    subplotter(4, 1, 3)
    C = linspecer(2)
    plot(t, np.real(z), color=C[0])
    plot(t, np.imag(z), color=C[1])
    legend(['x', 'y'])
    xlabel('time')
    nicefy(touch_limits=True)


    sac_ranges = list(zip(saccades_df['starts'], saccades_df['starts'] + saccades_df['durations']))
    plot_range(sac_ranges)


def save_and_close(path):
    plt.savefig(path, dpi=600)
    plt.close()


def create_map(location_name, address):
    geolocator = Nominatim(user_agent="my-application")
    location = geolocator.geocode('149 Clark Ave, Chelsea, MA')

    service = Static(
        access_token=
        'pk.eyJ1Ijoic2hhdW5wYXRlbCIsImEiOiJjazhkbjFiMnEwd2M3M2VzMjlvbXVmN3o4In0.-E-gYWZDXsW3zzpWJdRLcA'
    )

    feature = {
        'type': 'Feature',
        'properties': {
            'name': location_name
        },
        'geometry': {
            'type': 'Point',
            'coordinates': [location.longitude, location.latitude]
        }
    }

    response = service.image(
        'mapbox.streets',
        features=feature,
        lon=location.longitude,
        lat=location.latitude,
        z=12,
        retina=True)

    j = Image.open(io.BytesIO(response.content))

    plt.imshow(j)
    plt.axis('off')


def plot_visit_id_creation(summary_df):
    # plots when visit_ids have been created throughout the day

    fig_sizer(5,15)
    sns.set_style('white')
    _, bins, _ = plt.hist(summary_df.loc[summary_df['success'] == True, 'time_of_creation_EST'], bins=24, color='#0197F6', alpha=0.7,
                          label='successful')
    plt.hist(summary_df.loc[summary_df['success'] == False, 'time_of_creation_EST'], bins=bins, color='#D7263D', alpha=0.9,
             label='unsuccessful')

    nicefy()
    plt.title('Visit ID Creation Time (EST) Distribution')
    plt.grid(False)

    plt.xticks(rotation=90)
    plt.legend(loc=2)


def get_subj_control_colors():
    C = sns.color_palette('Set1', 1)
    C += [(.5, .5, .5)]
    return C



def plot_gaze(processed_df, visit_exam_id):
    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    processed_data = current_df.iloc[0]['processed']

    eye_df = processed_data['eye']

    plot_eye_speed(eye_df['z'], eye_df['t'], c=C[0], frac_c=.2)

    nicefy(fsize=10)

    return


# todo: move these where they belong
def plot_vor_xcorr(df):
    ds = np.median(df['Delta Time'])
    x = df['Camera Position X'].values
    y =  df['Left Eye Position X'].values
    [corrs, lags] = xcorr(x, y, ds)

    n=500
    plot(lags[n:-n], corrs[n:-n])
    print(lags[np.argmax(corrs)] * 1000)
    xlabel("Lags (s)")
    ylabel("Cross correlation")
    title("Correlation between head position and eye position")


def plot_vor(df):
    x = df['Camera Position X'].values
    # x2 = df['Dot2 Position X'].values
    y = df['Left Eye Position X'].values

    I = (x != 0) & (y != 0)
    #     fig, ax1 = plt.subplots()
    ax1 = gca()
    ax1.plot(x[I], color=C[0])
    #     ax1.plot(x2[I], color=C[2])
    ax1.set_ylabel('C Position X', color=C[0])
    #     legend(['Camera Position X'])

    ax2 = ax1.twinx()
    ax2.plot(y[I], color =C[1])
    ax2.set_ylabel('Left Eye Position X', color=C[1])
    ax1.set_xlabel('Time (s)')
    ax1.set_yticks([])
    nicefy()

