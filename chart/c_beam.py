from chart.c_general import *



def plot_beam(results):

    stim_df = results['stim_df']


    # thumb movements
    factor = 1
    fig_sizer(8, 12)

    for jj in range(3):
        exp_direction = ['LEFT', 'RIGHT', 'NONE'][jj]
        sub_df = stim_df[stim_df['Expected Thumb Text'] == exp_direction]

        if jj == 0:
            sub_df['reaction'] = sub_df['thumb z'].apply(lambda w: np.argmax(np.real(w) < -.1))
        else:
            sub_df['reaction'] = sub_df['thumb z'].apply(lambda w: np.argmax(np.real(w) > .1))

        sub_df = sub_df.sort_values('reaction')

        C = linspecer(len(sub_df), color='viridis')
        subplotter(2, 3, jj)
        title('expected direction: ' + exp_direction)

        for ii in range(len(sub_df)):
            plot(sub_df.iloc[ii]['time'], sub_df.iloc[ii]['thumb z'].real + factor * ii, color=C[ii])

        nicefy(touch_limits=True)
        plot_zero(lineheight=0, axx='y')
        xlim([-.3, 1.5])
        ylabel('thumb movement x')
        xlabel('time')
        #     ylim([0, 200])
        yticks([])



    # eye movements
    factor = 7

    for jj in range(3):
        exp_direction = ['LEFT', 'RIGHT', 'NONE'][jj]
        sub_df = stim_df[stim_df['Expected Thumb Text'] == exp_direction]
        if jj == 0:
            sub_df['reaction'] = sub_df['eye z'].apply(lambda w: np.argmax(np.real(w) < 7))
        else:
            sub_df['reaction'] = sub_df['eye z'].apply(lambda w: np.argmax(np.real(w) > 12.5))

        sub_df = sub_df.sort_values('reaction')

        C = linspecer(len(sub_df), color='viridis')
        subplotter(2, 3, jj + 3)
        title('expected direction: ' + exp_direction)

        for ii in range(len(sub_df)):
            plot(sub_df.iloc[ii]['time'], sub_df.iloc[ii]['eye z'].real + factor * ii, color=C[ii])

        nicefy(touch_limits=True)
        plot_zero(lineheight=0, axx='y')
        xlim([-.3, 1.5])
        ylabel('eye movement x')
        xlabel('time')
        yticks([])

    return 'png'
