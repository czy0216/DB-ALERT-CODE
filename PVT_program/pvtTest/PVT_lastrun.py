#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.4),
    on 五月 16, 2024, at 10:14
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.4'
expName = 'PVT'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    '姓名': '',
    '测试次数': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s' % (expInfo['姓名'], expInfo['测试次数'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='E:\\测试程序\\pvtTest\\PVT_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color='black', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'black'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('InstrucKey') is None:
        # initialise InstrucKey
        InstrucKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='InstrucKey',
        )
    if deviceManager.getDevice('dontrespond') is None:
        # initialise dontrespond
        dontrespond = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='dontrespond',
        )
    if deviceManager.getDevice('Response') is None:
        # initialise Response
        Response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Response',
        )
    if deviceManager.getDevice('GoodbyeEnd') is None:
        # initialise GoodbyeEnd
        GoodbyeEnd = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='GoodbyeEnd',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instructions" ---
    InstructionsText = visual.TextStim(win=win, name='InstructionsText',
        text='感谢您参与精神运动警觉性测试\n\n屏幕会在随机间隔（2-10秒）后出现数字，\n您需要在数字出现后尽快按下空格键。\n测试持续时间10分钟。\n\n请您准备好后，按下 空格键 开始测试',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='Arabic',
        depth=0.0);
    InstrucKey = keyboard.Keyboard(deviceName='InstrucKey')
    
    # --- Initialize components for Routine "ISI" ---
    # Run 'Begin Experiment' code from ISIcode
    # All the durations are in seconds
    minISI = 2
    maxISI = 10
    
    # Task duration (s)
    length_of_task = 600
    
    # Feedback duration ：按下后展示结果的时间
    feed = 0.5
    
    # A timer
    timing = core.Clock()
    
    # Loading the beep sound
    # warning_beep = sound.Sound('beep.wav')
    ISI_empty_text = visual.TextStim(win=win, name='ISI_empty_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    dontrespond = keyboard.Keyboard(deviceName='dontrespond')
    
    # --- Initialize components for Routine "Target" ---
    Targetstim = visual.TextStim(win=win, name='Targetstim',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color=[1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Response = keyboard.Keyboard(deviceName='Response')
    
    # --- Initialize components for Routine "Feedback" ---
    Feedback_text = visual.TextStim(win=win, name='Feedback_text',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "End_task" ---
    
    # --- Initialize components for Routine "The_end" ---
    Goodbye = visual.TextStim(win=win, name='Goodbye',
        text='感谢您的参与\n\n请按下 空格键 结束测试',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    GoodbyeEnd = keyboard.Keyboard(deviceName='GoodbyeEnd')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions.started', globalClock.getTime(format='float'))
    InstrucKey.keys = []
    InstrucKey.rt = []
    _InstrucKey_allKeys = []
    # keep track of which components have finished
    InstructionsComponents = [InstructionsText, InstrucKey]
    for thisComponent in InstructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *InstructionsText* updates
        
        # if InstructionsText is starting this frame...
        if InstructionsText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            InstructionsText.frameNStart = frameN  # exact frame index
            InstructionsText.tStart = t  # local t and not account for scr refresh
            InstructionsText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(InstructionsText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'InstructionsText.started')
            # update status
            InstructionsText.status = STARTED
            InstructionsText.setAutoDraw(True)
        
        # if InstructionsText is active this frame...
        if InstructionsText.status == STARTED:
            # update params
            pass
        
        # *InstrucKey* updates
        waitOnFlip = False
        
        # if InstrucKey is starting this frame...
        if InstrucKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            InstrucKey.frameNStart = frameN  # exact frame index
            InstrucKey.tStart = t  # local t and not account for scr refresh
            InstrucKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(InstrucKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'InstrucKey.started')
            # update status
            InstrucKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(InstrucKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(InstrucKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if InstrucKey.status == STARTED and not waitOnFlip:
            theseKeys = InstrucKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _InstrucKey_allKeys.extend(theseKeys)
            if len(_InstrucKey_allKeys):
                InstrucKey.keys = _InstrucKey_allKeys[-1].name  # just the last key pressed
                InstrucKey.rt = _InstrucKey_allKeys[-1].rt
                InstrucKey.duration = _InstrucKey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in InstructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
    # check responses
    if InstrucKey.keys in ['', [], None]:  # No response was made
        InstrucKey.keys = None
    thisExp.addData('InstrucKey.keys',InstrucKey.keys)
    if InstrucKey.keys != None:  # we had a response
        thisExp.addData('InstrucKey.rt', InstrucKey.rt)
        thisExp.addData('InstrucKey.duration', InstrucKey.duration)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    PVT_Trials = data.TrialHandler(nReps=150.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='PVT_Trials')
    thisExp.addLoop(PVT_Trials)  # add the loop to the experiment
    thisPVT_Trial = PVT_Trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPVT_Trial.rgb)
    if thisPVT_Trial != None:
        for paramName in thisPVT_Trial:
            globals()[paramName] = thisPVT_Trial[paramName]
    
    for thisPVT_Trial in PVT_Trials:
        currentLoop = PVT_Trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPVT_Trial.rgb)
        if thisPVT_Trial != None:
            for paramName in thisPVT_Trial:
                globals()[paramName] = thisPVT_Trial[paramName]
        
        # --- Prepare to start Routine "ISI" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ISI.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from ISIcode
        # ISI is then set each routine
        randISI = random() * (maxISI - minISI) + minISI
        
        # If it is the first trial
        if PVT_Trials.thisN == 0:
            overall_timer = core.Clock()#记录刚刚开始实验的时间（第一次）
            realISI = 0
            
        if PVT_Trials.thisN > 0:
            # We count the duration of the feedback as part of the ISI
            realISI = feed
        
        # A message when participant miss
        message = 'You did not hit the button!'
        
        # Adding the ISI so it is saved in the datafile ISI记录randISI时间
        thisExp.addData('ISI', randISI)
        dontrespond.keys = []
        dontrespond.rt = []
        _dontrespond_allKeys = []
        # keep track of which components have finished
        ISIComponents = [ISI_empty_text, dontrespond]
        for thisComponent in ISIComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ISI" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from ISIcode
            keys = dontrespond.getKeys(keyList=['space'], waitRelease=False)
            keys = [key.name for key in keys]
            
            # Append True to list if a key is pressed, clear list if not
            if "space" in keys:
                 message = "Too soon!"
                 continueRoutine = False
                
            
            # *ISI_empty_text* updates
            
            # if ISI_empty_text is starting this frame...
            if ISI_empty_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ISI_empty_text.frameNStart = frameN  # exact frame index
                ISI_empty_text.tStart = t  # local t and not account for scr refresh
                ISI_empty_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI_empty_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ISI_empty_text.started')
                # update status
                ISI_empty_text.status = STARTED
                ISI_empty_text.setAutoDraw(True)
            
            # if ISI_empty_text is active this frame...
            if ISI_empty_text.status == STARTED:
                # update params
                pass
            
            # if ISI_empty_text is stopping this frame...
            if ISI_empty_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ISI_empty_text.tStartRefresh + randISI-realISI-frameTolerance:
                    # keep track of stop time/frame for later
                    ISI_empty_text.tStop = t  # not accounting for scr refresh
                    ISI_empty_text.tStopRefresh = tThisFlipGlobal  # on global time
                    ISI_empty_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ISI_empty_text.stopped')
                    # update status
                    ISI_empty_text.status = FINISHED
                    ISI_empty_text.setAutoDraw(False)
            
            # *dontrespond* updates
            waitOnFlip = False
            
            # if dontrespond is starting this frame...
            if dontrespond.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dontrespond.frameNStart = frameN  # exact frame index
                dontrespond.tStart = t  # local t and not account for scr refresh
                dontrespond.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dontrespond, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dontrespond.started')
                # update status
                dontrespond.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(dontrespond.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(dontrespond.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if dontrespond is stopping this frame...
            if dontrespond.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dontrespond.tStartRefresh + randISI-1-frameTolerance:
                    # keep track of stop time/frame for later
                    dontrespond.tStop = t  # not accounting for scr refresh
                    dontrespond.tStopRefresh = tThisFlipGlobal  # on global time
                    dontrespond.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dontrespond.stopped')
                    # update status
                    dontrespond.status = FINISHED
                    dontrespond.status = FINISHED
            if dontrespond.status == STARTED and not waitOnFlip:
                theseKeys = dontrespond.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _dontrespond_allKeys.extend(theseKeys)
                if len(_dontrespond_allKeys):
                    dontrespond.keys = _dontrespond_allKeys[-1].name  # just the last key pressed
                    dontrespond.rt = _dontrespond_allKeys[-1].rt
                    dontrespond.duration = _dontrespond_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ISIComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ISI" ---
        for thisComponent in ISIComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ISI.stopped', globalClock.getTime(format='float'))
        # check responses
        if dontrespond.keys in ['', [], None]:  # No response was made
            dontrespond.keys = None
        PVT_Trials.addData('dontrespond.keys',dontrespond.keys)
        if dontrespond.keys != None:  # we had a response
            PVT_Trials.addData('dontrespond.rt', dontrespond.rt)
            PVT_Trials.addData('dontrespond.duration', dontrespond.duration)
        # the Routine "ISI" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Target" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Target.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from Target_code
        # Reset the timer
        timing.reset()
        
        # Check for response
        if message == 'Too soon!':
            # Adding 0 to Accuracy and missing to RTms
            thisExp.addData('Accuracy', 0)
            thisExp.addData('RTms', np.NAN)
            # End the Routine to continue next trial
            continueRoutine = False
        
        Response.keys = []
        Response.rt = []
        _Response_allKeys = []
        # keep track of which components have finished
        TargetComponents = [Targetstim, Response]
        for thisComponent in TargetComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Target" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from Target_code
            # counter in seconds
            time = int(round(timing.getTime(), 3) * 1000)
            
            
            
            # *Targetstim* updates
            
            # if Targetstim is starting this frame...
            if Targetstim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Targetstim.frameNStart = frameN  # exact frame index
                Targetstim.tStart = t  # local t and not account for scr refresh
                Targetstim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Targetstim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Targetstim.started')
                # update status
                Targetstim.status = STARTED
                Targetstim.setAutoDraw(True)
            
            # if Targetstim is active this frame...
            if Targetstim.status == STARTED:
                # update params
                Targetstim.setText(time, log=False)
            
            # if Targetstim is stopping this frame...
            if Targetstim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Targetstim.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    Targetstim.tStop = t  # not accounting for scr refresh
                    Targetstim.tStopRefresh = tThisFlipGlobal  # on global time
                    Targetstim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Targetstim.stopped')
                    # update status
                    Targetstim.status = FINISHED
                    Targetstim.setAutoDraw(False)
            
            # *Response* updates
            waitOnFlip = False
            
            # if Response is starting this frame...
            if Response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Response.frameNStart = frameN  # exact frame index
                Response.tStart = t  # local t and not account for scr refresh
                Response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Response.started')
                # update status
                Response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(Response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if Response is stopping this frame...
            if Response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Response.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    Response.tStop = t  # not accounting for scr refresh
                    Response.tStopRefresh = tThisFlipGlobal  # on global time
                    Response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Response.stopped')
                    # update status
                    Response.status = FINISHED
                    Response.status = FINISHED
            if Response.status == STARTED and not waitOnFlip:
                theseKeys = Response.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _Response_allKeys.extend(theseKeys)
                if len(_Response_allKeys):
                    Response.keys = _Response_allKeys[-1].name  # just the last key pressed
                    Response.rt = _Response_allKeys[-1].rt
                    Response.duration = _Response_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TargetComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Target" ---
        for thisComponent in TargetComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Target.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from Target_code
        if type(Response.rt) is float:#正确情况下
            message = str(round(Response.rt * 1000))
            thisExp.addData('Accuracy', 1)
            thisExp.addData('RTms', Response.rt * 1000)#反应时间
            
        # PsychoPy is not running the trial for more than 29.991...超时
        if timing.getTime() >= 29.99:
                message = 'No response!'
                warning_beep.play()
                Response.rt = timing.getTime()
                thisExp.addData('RTms', np.NAN)
                thisExp.addData('Accuracy', 0)
                continueRoutine = False
        # check responses
        if Response.keys in ['', [], None]:  # No response was made
            Response.keys = None
        PVT_Trials.addData('Response.keys',Response.keys)
        if Response.keys != None:  # we had a response
            PVT_Trials.addData('Response.rt', Response.rt)
            PVT_Trials.addData('Response.duration', Response.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "Feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Feedback.started', globalClock.getTime(format='float'))
        Feedback_text.setText(message)
        # keep track of which components have finished
        FeedbackComponents = [Feedback_text]
        for thisComponent in FeedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Feedback_text* updates
            
            # if Feedback_text is starting this frame...
            if Feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Feedback_text.frameNStart = frameN  # exact frame index
                Feedback_text.tStart = t  # local t and not account for scr refresh
                Feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Feedback_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Feedback_text.started')
                # update status
                Feedback_text.status = STARTED
                Feedback_text.setAutoDraw(True)
            
            # if Feedback_text is active this frame...
            if Feedback_text.status == STARTED:
                # update params
                pass
            
            # if Feedback_text is stopping this frame...
            if Feedback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Feedback_text.tStartRefresh + feed-frameTolerance:
                    # keep track of stop time/frame for later
                    Feedback_text.tStop = t  # not accounting for scr refresh
                    Feedback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    Feedback_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Feedback_text.stopped')
                    # update status
                    Feedback_text.status = FINISHED
                    Feedback_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FeedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Feedback" ---
        for thisComponent in FeedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Feedback.stopped', globalClock.getTime(format='float'))
        # the Routine "Feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "End_task" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('End_task.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        End_taskComponents = []
        for thisComponent in End_taskComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "End_task" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in End_taskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "End_task" ---
        for thisComponent in End_taskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('End_task.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from End_task_2
        # Get the time in the task
        time_in_task = overall_timer.getTime()
        
        # If time_in_task corresponds to the duration we set previously we end te task结束循环
        if time_in_task >= length_of_task:
            continueRoutine = False
            PVT_Trials.finished = True
        
        # the Routine "End_task" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 150.0 repeats of 'PVT_Trials'
    
    # get names of stimulus parameters
    if PVT_Trials.trialList in ([], [None], None):
        params = []
    else:
        params = PVT_Trials.trialList[0].keys()
    # save data for this loop
    PVT_Trials.saveAsExcel(filename + '.xlsx', sheetName='PVT_Trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "The_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('The_end.started', globalClock.getTime(format='float'))
    GoodbyeEnd.keys = []
    GoodbyeEnd.rt = []
    _GoodbyeEnd_allKeys = []
    # keep track of which components have finished
    The_endComponents = [Goodbye, GoodbyeEnd]
    for thisComponent in The_endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "The_end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Goodbye* updates
        
        # if Goodbye is starting this frame...
        if Goodbye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Goodbye.frameNStart = frameN  # exact frame index
            Goodbye.tStart = t  # local t and not account for scr refresh
            Goodbye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Goodbye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Goodbye.started')
            # update status
            Goodbye.status = STARTED
            Goodbye.setAutoDraw(True)
        
        # if Goodbye is active this frame...
        if Goodbye.status == STARTED:
            # update params
            pass
        
        # *GoodbyeEnd* updates
        waitOnFlip = False
        
        # if GoodbyeEnd is starting this frame...
        if GoodbyeEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            GoodbyeEnd.frameNStart = frameN  # exact frame index
            GoodbyeEnd.tStart = t  # local t and not account for scr refresh
            GoodbyeEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(GoodbyeEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'GoodbyeEnd.started')
            # update status
            GoodbyeEnd.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(GoodbyeEnd.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(GoodbyeEnd.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if GoodbyeEnd.status == STARTED and not waitOnFlip:
            theseKeys = GoodbyeEnd.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _GoodbyeEnd_allKeys.extend(theseKeys)
            if len(_GoodbyeEnd_allKeys):
                GoodbyeEnd.keys = _GoodbyeEnd_allKeys[-1].name  # just the last key pressed
                GoodbyeEnd.rt = _GoodbyeEnd_allKeys[-1].rt
                GoodbyeEnd.duration = _GoodbyeEnd_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in The_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "The_end" ---
    for thisComponent in The_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('The_end.stopped', globalClock.getTime(format='float'))
    # check responses
    if GoodbyeEnd.keys in ['', [], None]:  # No response was made
        GoodbyeEnd.keys = None
    thisExp.addData('GoodbyeEnd.keys',GoodbyeEnd.keys)
    if GoodbyeEnd.keys != None:  # we had a response
        thisExp.addData('GoodbyeEnd.rt', GoodbyeEnd.rt)
        thisExp.addData('GoodbyeEnd.duration', GoodbyeEnd.duration)
    thisExp.nextEntry()
    # the Routine "The_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
