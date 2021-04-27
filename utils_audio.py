####################################################################
# FUNCTIONS TO PROCESS AUDIO
####################################################################

# IMPORTING LIBRARIES
import boto3
import numpy as np
import IPython.display
import librosa
import librosa.display
import io
from pydub import AudioSegment
import scipy.signal as signal
from google.cloud import speech_v1
import os
import csv
import parselmouth
from parselmouth.praat import call
import logging

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./.aws/speech-api-google-credentials.json"


# ----------------------------------------------------------------
# CONVERTING AUDIO DATA FROM array.array TO np.ndarray
# ----------------------------------------------------------------
def audio_to_ndarray(audio):
    
    if type(audio) != np.ndarray:
        
        # convert to np.ndarray
        audio = np.asarray(audio).astype(float)
        
    return audio

# ----------------------------------------------------------------
# FUNCITON TO CALCULATE SPEECH ONSET
# ----------------------------------------------------------------
def speech_onset(audio, sr, threshold = 0):
    
    # converting audio
    audio = audio_to_ndarray(audio)
    
    # tracking energy in speech 
    onset_tracker = librosa.onset.onset_strength(y=audio, sr=sr)

    # determining corresponding time stamps (x_values)
    times = librosa.times_like(onset_tracker, sr=sr)

    # determining energy differential peaks (i.e. onset of speech)
    onset_peaks = librosa.onset.onset_detect(y=audio, sr=sr)
    
    # saving peaks that are above a threshold
    peak_max = 0
    peaks_thresh = []
    for peak in onset_peaks:
        if onset_tracker[peak] > threshold:
            peaks_thresh.append(peak)
        
        if onset_tracker[peak] > onset_tracker[peak_max]:
            peak_max = peak
            
    if peaks_thresh == []:
        onset_peaks = [peak_max]
    else:
        onset_peaks = peaks_thresh

    return times, onset_tracker, onset_peaks

# ----------------------------------------------------------------
# FUNCTION TO COMPUTE SPECTROGRAM
# ----------------------------------------------------------------
def compute_spectrogram(y, sr):
    
    # grabbing first 5 seconds of audio
    # start  = start_sec * sr
    # finish = finish_sec * sr
    # y_sub = y[start:finish]
    
    # converting audio
    y = audio_to_ndarray(y)

    # ----------------------------------------------------------------
    # COMPUTUING MEL SPECTROGRAM VALUES
    # ----------------------------------------------------------------
    # This is an STFT calculated over a:
    # 1. [n_fft] 20ms hanning window,
    n_fft = int(sr * 0.02)

    # 2. [hop_length] 10ms stride/step size
    hop_length = int(sr * 0.01)

    # 3. [power] then squared
    power = 2

    # 4. [n_mels] then binned over 128 mel sized triangular bins
    n_mels = 128

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000, power=power,
                                       hop_length=hop_length, n_fft=n_fft)


    # Converting Power to DB
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB


# ----------------------------------------------------------------
# FUNCTION TO TRANSCRIBE SPEECH TO TEXT USING GOOGLE API
#
#  uri:      is the s3 bucket location of the audio .wav file
#  savefile: is the path where the transcripts will be saved to.
# ----------------------------------------------------------------
def transcribe_speech_to_text_with_word_timestamps(audio, savefile='tmp.transcript.csv', audio_type='long'):
    """
    Print start and end time of each word spoken in audio file from Cloud Storage

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    # -------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------
    client = speech_v1.SpeechClient()

    # When enabled, the first result returned by the API will include a list
    # of words and the start and end time offsets (timestamps) for those words.
    enable_word_time_offsets = True

    # The language of the supplied audio
    language_code = "en-US"

    config = {  "enable_word_time_offsets": enable_word_time_offsets,
                "language_code": language_code}
    
    # -------------------------------------------------------------------
    # READING AUDIO FILE
    # -------------------------------------------------------------------
    # in milliseconds
    start = 0 * 1000
    finish = 59 * 1000

    
    # grabbing first 60 seconds 
    # (otherwise google-speech expects the audio to be stored on their cloud for processing)
    if len(audio) < finish:
        finish = len(audio)
    
    else:
        audio = audio[start:finish]
        
    # simple export of file (error when passing it directly to google-speech)
    file_handle = audio.export("tmp.wav", format="wav")
    
    # read file
    with io.open("tmp.wav", "rb") as f:
        content = f.read()
    os.remove("tmp.wav")
    
    # define as content
    audio = {"content": content}

    # -------------------------------------------------------------------
    # TRANSCRIBING
    # -------------------------------------------------------------------
    
    # IF LONG RUNNING AUDIO IS CALLED TO TRANSCRIBE
    if audio_type == 'long':
        operation = client.long_running_recognize(config, audio)
        
        print("Waiting for operation to complete...")
        response = operation.result()

        
    # IF LONG RUNNING AUDIO IS CALLED TO TRANSCRIBE
    elif audio_type == 'short':
        response = client.recognize(config, audio)

    # LOOPING THROUGH EACH TRANSCRIPT SEGMENT
    transcripts = []
    for result in response.results:
        
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        # print(u"Transcript: {}".format(alternative.transcript))
    
        # -------------------------------------------------------------------
        # PRINTING TO SCREEN
        # -------------------------------------------------------------------
        # Print the start and end time of each word
        for word in alternative.words:

            start_time = word.start_time.seconds + word.start_time.nanos / 1000000000
            end_time   = word.end_time.seconds + word.end_time.nanos / 1000000000
            # print(u"Word: {}".format(word.word))
            # print("Start: {} secs | End: {} secs \t Word: {}".format(
            #         start_time,
            #         end_time,
            #         word.word ))

            word_dict = {'time_start_seconds': start_time, 'transcript': word.word, 'time_end_seconds': end_time}
            transcripts.append(word_dict)

    # -------------------------------------------------------------------
    # SAVING TRANSCRIPTION
    # -------------------------------------------------------------------
    if len(transcripts) > 0:
        write_list_of_dicts_to_csv(transcripts, savefile)
        print(' Finished writing to file:', savefile)
    else:
        logging.warning("Audio file has no transcripts.")
    
    return response
    
# -------------------------------------------------------------------
# FUNCTION TO WRITE LIST OF DICTS TO CSV FILE
# -------------------------------------------------------------------
# todo: replace this with a line in pandas
def write_list_of_dicts_to_csv(list_of_dicts, filename):

    # writing results to file
    keys = list_of_dicts[0].keys()
    with open(filename, "w") as f:
        dict_writer = csv.DictWriter(f, keys, delimiter=",")
        dict_writer.writeheader()
        for row in list_of_dicts:
            dict_writer.writerow(row)

# --------------------------------------------------------------------
# CALCULATING WORD ERROR RATE, INSERTIONS, SUBSTITUTION, DELETIONS.
# --------------------------------------------------------------------
def calculate_word_error_rate(ref, hyp, debug=False):
    
    r = ref.lower().split()
    h = hyp.lower().split()
    
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    
    SUB_PENALTY = 1
    INS_PENALTY = 1
    DEL_PENALTY = 1
     
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
         
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
     
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                 
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
        return (numSub + numDel + numIns) / (float) (len(r))
    
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER': wer_result, 'Cor': numCor, 'Sub': numSub, 'Ins': numIns, 'Del': numDel}


# --------------------------------------------------------------------
# CALCULATING JITTER
# --------------------------------------------------------------------
def calculate_jitter(snd, statistic_type='local', window=0.25, step=0.01):
    """        
    # the shortest possible interval that will be used in the computation of jitter, in seconds. 
    # If an interval is shorter than this, it will be ignored in the computation of jitter 
    # (and the previous and next intervals will not be regarded as consecutive). 
    # This setting will normally be very small, say 0.1 ms.
    period_floor   = 0.0001

    # the longest possible interval that will be used in the computation of jitter, in seconds. 
    # If an interval is longer than this, it will be ignored in the computation of jitter 
    # (and the previous and next intervals will not be regarded as consecutive). 
    # For example, if the minimum frequency of periodicity is 50 Hz, set this setting to 0.02 seconds; 
    # intervals longer than that could be regarded as voiceless stretches and will be ignored in the computation.
    period_ceiling = 0.02

    # the largest possible difference between consecutive intervals that will be used in the computation of jitter. 
    # If the ratio of the durations of two consecutive intervals is greater than this, this pair of intervals will 
    # be ignored in the computation of jitter (each of the intervals could still take part in the computation of 
    # jitter in a comparison with its neighbour on the other side).
    max_period_factor = 1.3

    # window in seconds
    window = 0.5

    # window shift in seconds
    step = 0.01
    
    # if t_start = 0 and t_end = 0, full waveform is used
    
    Reference for Jitter: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html
    Reference for Jitter Eqn: https://core.ac.uk/download/pdf/82375396.pdf
    """
    
    # statistic = {'jitter': {'period_floor': 0.0001,'period_ceiling': 0.02, 'max_period_factor':1.3}}
    period_floor      = 0.0001
    period_ceiling    = 0.02
    max_period_factor = 1.6

    # GET PITCH
    pitch          = snd.to_pitch()
    
    # GET INTERVALS AT WHICH PITCH CROSSES ZERO
    pointProcess   = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

    # allowed types: 'local', 'local,absolute', 'rap', 'ppq5', 'ddp'
    praat_cmd = "Get jitter ({statistic_type})".format(statistic_type = statistic_type)

    # ----------------------------------------------------------------
    # LOOP THROUGH WINDOWS TO EVALUATE JITTER
    # ----------------------------------------------------------------
    jitter, timestamps = [], []
    for t_start in np.arange(0, snd.duration-window, step):

        # if t_start = 0 and t_end = 0, full waveform is used
        t_end = t_start + window

        # CALCULATE JITTER OVER WINDOW
        jitt  = call(pointProcess, praat_cmd, t_start, t_end, period_floor, period_ceiling, max_period_factor)

        timestamps.append(t_end)
        jitter.append(jitt)
        # break

    return jitter, timestamps


# --------------------------------------------------------------------
# CALCULATING SHIMMER
# --------------------------------------------------------------------
def calculate_shimmer(snd, statistic_type='local', window=0.25, step=0.01):
    """        
    # the shortest possible interval that will be used in the computation of shimmer, in seconds. 
    # If an interval is shorter than this, it will be ignored in the computation of shimmer 
    # (and the previous and next intervals will not be regarded as consecutive). 
    # This setting will normally be very small, say 0.1 ms.
    period_floor   = 0.0001

    # the longest possible interval that will be used in the computation of shimmer, in seconds. 
    # If an interval is longer than this, it will be ignored in the computation of shimmer 
    # (and the previous and next intervals will not be regarded as consecutive). 
    # For example, if the minimum frequency of periodicity is 50 Hz, set this setting to 0.02 seconds; 
    # intervals longer than that could be regarded as voiceless stretches and will be ignored in the computation.
    period_ceiling = 0.02

    # the largest possible difference between consecutive intervals that will be used in the computation of shimmer. 
    # If the ratio of the durations of two consecutive intervals is greater than this, this pair of intervals will 
    # be ignored in the computation of shimmer (each of the intervals could still take part in the computation of 
    # shimmer in a comparison with its neighbour on the other side).
    max_period_factor = 1.3

    # window in seconds
    window = 0.5

    # window shift in seconds
    step = 0.01
    
    # if t_start = 0 and t_end = 0, full waveform is used
    
    Reference for Shimmer: http://www.fon.hum.uva.nl/praat/manual/Voice_3__Shimmer.html
    Reference for Shimmer Eqn: https://core.ac.uk/download/pdf/82375396.pdf
    """
    
    # statistic = {'shimmer': {'period_floor': 0.0001,'period_ceiling': 0.02, 'max_period_factor':1.3}}
    period_floor      = 0.0001
    period_ceiling    = 0.02
    max_period_factor = 1.6

    # GET PITCH
    pitch          = snd.to_pitch()
    
    # GET INTERVALS AT WHICH PITCH CROSSES ZERO
    pointProcess   = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

    # allowed types: 'local', 'local_dB', 'apq3', 'apq5', 'apq11', 'dda'
    praat_cmd = "Get shimmer ({statistic_type})".format(statistic_type = statistic_type)

    # ----------------------------------------------------------------
    # LOOP THROUGH WINDOWS TO EVALUATE SHIMMER
    # ----------------------------------------------------------------
    shimmer, timestamps = [], []
    for t_start in np.arange(0,snd.duration-window,step):

        # if t_start = 0 and t_end = 0, full waveform is used
        t_end = t_start + window

        # CALCULATE SHIMMER OVER WINDOW
        shimm  = call([snd, pointProcess], praat_cmd, t_start, t_end, period_floor, period_ceiling, max_period_factor, 1.6)

        timestamps.append(t_end)
        shimmer.append(shimm)
        # break

    return shimmer, timestamps


# -------------------------------------------------------------------------------
# CALCULATE HARMONIC-TO-NOISE RATIO
# -------------------------------------------------------------------------------
def calculate_HNR(snd, step=0.01, pitch_min=75, silence_threshold=0.05, number_periods_min=4.5):
        
    # CALCULATE HARMONIC-TO-NOISE RATIO
    hnr = call(snd, "To Harmonicity (cc)", step, pitch_min, silence_threshold, number_periods_min)
        
    return hnr


#################################################################################
# SAVE FBANK/MFCC FEATURES
#################################################################################
def save_spectral_features(spec, header, filename):
    """
    
    Input: 
        spec:     <numpy.array> 
                  can be any N-dim time-series generated by librosa.
                  e.g. fbank, mfcc, intensity, zero-crossing-rate
                  must be passed as n_examples (rows) x n_features (columns)
                
        header:   <str> 
                  of column name prefix that will be saved.
        
    Output:
        filename: <str>
                  path to file where data will be saved.  
    """
    
    
    # CALCULATING TIME INDEX
    # step size of shift in window
    step = 0.01
    t_seconds = np.arange(0, spec.shape[0] * step, step)

    # OPENING FILE FOR SAVING
    with open(filename, 'w') as f:
        
        # DEFINING HEADER OF CSV FILE
        num_columns = spec.shape[1]
        headers = [header + str(i) for i in range(num_columns)]
        headers = ','.join(headers) + '\n'
        headers = 't_seconds,' + headers
        
        # WRITING HEADER TO FILE
        f.write(headers)

        # COMBINING TIME INDEX AND FEATURES
        data = np.vstack((t_seconds,spec.T)).T
        
        # SAVING TO FILE
        np.savetxt(f, data, delimiter=",")
        
#################################################################################
# SAVE LIBROSA SPEECH/SILENCE SPLIT
#################################################################################
def save_split_features(feature, header, filename):
    
    # CALCULATING TIME INDEX
    # step size of shift in window
    # step = 0.01
    # t_seconds = np.arange(0, spec.shape[0] * step, step)

    # OPENING FILE FOR SAVING
    with open(filename, 'w') as f:
        
        # DEFINING HEADER OF CSV FILE
        num_columns = feature.shape[1]
        headers = [header + str(i) for i in range(num_columns)]
        headers = ','.join(headers) + '\n'
        # headers = 't_seconds,' + headers
        
        # WRITING HEADER TO FILE
        f.write(headers)

        # COMBINING TIME INDEX AND FEATURES
        # data = np.vstack((t_seconds,spec.T)).T
        
        # SAVING TO FILE
        np.savetxt(f, feature, delimiter=",")

#################################################################################
# SAVE PARSELMOUTH FEATURE
#################################################################################
def save_parselmouth_features(feature, t_seconds, header, filename):
    
    # print(feature.shape, t_seconds.shape)

    # OPENING FILE FOR SAVING
    with open(filename, 'w') as f:

        # DEFINING HEADER OF CSV FILE
        num_columns = feature.shape[1]
        headers = [header + str(i) for i in range(num_columns)]
        headers = ','.join(headers) + '\n'
        headers = 't_seconds,' + headers

        # WRITING HEADER TO FILE
        f.write(headers)

        # COMBINING TIME INDEX AND FEATURES
        data = np.hstack((t_seconds,feature))
        # print(data.shape)

        # SAVING TO FILE
        np.savetxt(f, data, delimiter=",")
        

def pass_filter(y, fs, filter_type, cutoff):
    """

    :param y: signal to filter
    :param type: type of filter
    :param fs: sampling rate
    :param cutoff: frequencies to filter by; one value for lp and hp, array for bp
    :return:
    """
    if filter_type == 'highpass':
        fil = signal.firwin(255, cutoff, pass_zero='highpass', fs=fs)
    elif filter_type == 'lowpass':
        fil = signal.firwin(255, cutoff, pass_zero='lowpass', fs=fs)
    elif filter_type == 'bandpass':
        fil = signal.firwin(255, [cutoff[0], cutoff[1]], pass_zero='bandpass', fs=fs)

    filtered = signal.filtfilt(fil, 1, y, axis=0)
    return filtered