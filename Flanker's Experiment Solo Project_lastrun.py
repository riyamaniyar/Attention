#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.3),
    on Fri Oct 16 10:53:12 2026
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
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.3'
expName = "Flanker's Experiment Solo Project"  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
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
_fullScr = False
_winSize = [1200, 750]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath="/Users/shradulrajput/Desktop/solo proj-flankers/Flanker's Experiment Solo Project_lastrun.py",
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
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
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='lms',
            backgroundImage='', backgroundFit='fill',
            blendMode='avg', useFBO=True,
            units='pix',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'lms'
        win.backgroundImage = ''
        win.backgroundFit = 'fill'
        win.units = 'pix'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
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
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
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
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


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
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
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
    
    # --- Initialize components for Routine "Intstructions" ---
    text = visual.TextStim(win=win, name='text',
        text='Welcome, and thank you for taking part.\n\n\n\nIn this task, a shape will appear in the centre of the screen.\nSometimes other shapes will appear on either side of it.\n\nYour job is to respond ONLY to the centre shape and ignore the others.\n\n\nIf the centre shape is  ▲  or  ●   press the  A  key.\n\nIf the centre shape is  ■  or  ◆   press the  L  key.\n\n\nKeep your eyes on the + in the middle of the screen.\n\nRespond as quickly and as accurately as you can.\n\n\n\n\nPress SPACE to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "Fixation" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Trial" ---
    Target = visual.TextStim(win=win, name='Target',
        text='',
        font='Arial',
        units='height', pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Flank1L = visual.TextStim(win=win, name='Flank1L',
        text='',
        font='Arial',
        units='height', pos=[0,0], draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    flank2L = visual.TextStim(win=win, name='flank2L',
        text='',
        font='Arial',
        units='height', pos=[0,0], draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    flank3L = visual.TextStim(win=win, name='flank3L',
        text='',
        font='Arial',
        units='height', pos=[0,0], draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    flank1R = visual.TextStim(win=win, name='flank1R',
        text='',
        font='Arial',
        units='height', pos=[0,0], draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    flank2R = visual.TextStim(win=win, name='flank2R',
        text='',
        font='Arial',
        units='height', pos=[0,0], draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    flank3R = visual.TextStim(win=win, name='flank3R',
        text='',
        font='Arial',
        units='height', pos=[0,0], draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "iti" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "practice_end" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Practice complete. The main task works the same way, but there is no feedback after each trial. Press SPACE to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
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
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Intstructions" ---
    # create an object to store info about Routine Intstructions
    Intstructions = data.Routine(
        name='Intstructions',
        components=[text, key_resp],
    )
    Intstructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Intstructions
    Intstructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intstructions.tStart = globalClock.getTime(format='float')
    Intstructions.status = STARTED
    thisExp.addData('Intstructions.started', Intstructions.tStart)
    Intstructions.maxDuration = None
    # keep track of which components have finished
    IntstructionsComponents = Intstructions.components
    for thisComponent in Intstructions.components:
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
    
    # --- Run Routine "Intstructions" ---
    thisExp.currentRoutine = Intstructions
    Intstructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            key_resp.clock.reset()  # now t=0
        if key_resp.status == STARTED:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Intstructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Intstructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Intstructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Intstructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intstructions" ---
    for thisComponent in Intstructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intstructions
    Intstructions.tStop = globalClock.getTime(format='float')
    Intstructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intstructions.stopped', Intstructions.tStop)
    thisExp.nextEntry()
    # the Routine "Intstructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Untitled spreadsheet.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Fixation" ---
        # create an object to store info about Routine Fixation
        Fixation = data.Routine(
            name='Fixation',
            components=[polygon],
        )
        Fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fixation
        Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fixation.tStart = globalClock.getTime(format='float')
        Fixation.status = STARTED
        thisExp.addData('Fixation.started', Fixation.tStart)
        Fixation.maxDuration = None
        # keep track of which components have finished
        FixationComponents = Fixation.components
        for thisComponent in Fixation.components:
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
        
        # --- Run Routine "Fixation" ---
        thisExp.currentRoutine = Fixation
        Fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Fixation,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Fixation.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Fixation.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation" ---
        for thisComponent in Fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fixation
        Fixation.tStop = globalClock.getTime(format='float')
        Fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fixation.stopped', Fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fixation.maxDurationReached:
            routineTimer.addTime(-Fixation.maxDuration)
        elif Fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Trial" ---
        # create an object to store info about Routine Trial
        Trial = data.Routine(
            name='Trial',
            components=[Target, Flank1L, flank2L, flank3L, flank1R, flank2R, flank3R, key_resp_2],
        )
        Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Target.setText(targetChar)
        Flank1L.setPos((flank1LX, 0))
        Flank1L.setText(flank1LChar)
        flank2L.setPos((flank2LX, 0))
        flank2L.setText(flank2LChar)
        flank3L.setPos((flank3LX, 0))
        flank3L.setText(flank3LChar)
        flank1R.setPos((flank1RX, 0))
        flank1R.setText(flank1RChar)
        flank2R.setPos((flank2RX, 0))
        flank2R.setText(flank2RChar)
        flank3R.setPos((flank3RX, 0))
        flank3R.setText(flank3RChar)
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # store start times for Trial
        Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Trial.tStart = globalClock.getTime(format='float')
        Trial.status = STARTED
        thisExp.addData('Trial.started', Trial.tStart)
        Trial.maxDuration = None
        # keep track of which components have finished
        TrialComponents = Trial.components
        for thisComponent in Trial.components:
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
        
        # --- Run Routine "Trial" ---
        thisExp.currentRoutine = Trial
        Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.7:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Target* updates
            
            # if Target is starting this frame...
            if Target.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                Target.frameNStart = frameN  # exact frame index
                Target.tStart = t  # local t and not account for scr refresh
                Target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Target.started')
                # update status
                Target.status = STARTED
                Target.setAutoDraw(True)
            
            # if Target is active this frame...
            if Target.status == STARTED:
                # update params
                pass
            
            # if Target is stopping this frame...
            if Target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Target.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Target.tStop = t  # not accounting for scr refresh
                    Target.tStopRefresh = tThisFlipGlobal  # on global time
                    Target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Target.stopped')
                    # update status
                    Target.status = FINISHED
                    Target.setAutoDraw(False)
            
            # *Flank1L* updates
            
            # if Flank1L is starting this frame...
            if Flank1L.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                Flank1L.frameNStart = frameN  # exact frame index
                Flank1L.tStart = t  # local t and not account for scr refresh
                Flank1L.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Flank1L, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Flank1L.started')
                # update status
                Flank1L.status = STARTED
                Flank1L.setAutoDraw(True)
            
            # if Flank1L is active this frame...
            if Flank1L.status == STARTED:
                # update params
                pass
            
            # if Flank1L is stopping this frame...
            if Flank1L.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Flank1L.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Flank1L.tStop = t  # not accounting for scr refresh
                    Flank1L.tStopRefresh = tThisFlipGlobal  # on global time
                    Flank1L.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Flank1L.stopped')
                    # update status
                    Flank1L.status = FINISHED
                    Flank1L.setAutoDraw(False)
            
            # *flank2L* updates
            
            # if flank2L is starting this frame...
            if flank2L.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                flank2L.frameNStart = frameN  # exact frame index
                flank2L.tStart = t  # local t and not account for scr refresh
                flank2L.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(flank2L, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'flank2L.started')
                # update status
                flank2L.status = STARTED
                flank2L.setAutoDraw(True)
            
            # if flank2L is active this frame...
            if flank2L.status == STARTED:
                # update params
                pass
            
            # if flank2L is stopping this frame...
            if flank2L.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > flank2L.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    flank2L.tStop = t  # not accounting for scr refresh
                    flank2L.tStopRefresh = tThisFlipGlobal  # on global time
                    flank2L.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'flank2L.stopped')
                    # update status
                    flank2L.status = FINISHED
                    flank2L.setAutoDraw(False)
            
            # *flank3L* updates
            
            # if flank3L is starting this frame...
            if flank3L.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                flank3L.frameNStart = frameN  # exact frame index
                flank3L.tStart = t  # local t and not account for scr refresh
                flank3L.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(flank3L, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'flank3L.started')
                # update status
                flank3L.status = STARTED
                flank3L.setAutoDraw(True)
            
            # if flank3L is active this frame...
            if flank3L.status == STARTED:
                # update params
                pass
            
            # if flank3L is stopping this frame...
            if flank3L.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > flank3L.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    flank3L.tStop = t  # not accounting for scr refresh
                    flank3L.tStopRefresh = tThisFlipGlobal  # on global time
                    flank3L.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'flank3L.stopped')
                    # update status
                    flank3L.status = FINISHED
                    flank3L.setAutoDraw(False)
            
            # *flank1R* updates
            
            # if flank1R is starting this frame...
            if flank1R.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                flank1R.frameNStart = frameN  # exact frame index
                flank1R.tStart = t  # local t and not account for scr refresh
                flank1R.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(flank1R, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'flank1R.started')
                # update status
                flank1R.status = STARTED
                flank1R.setAutoDraw(True)
            
            # if flank1R is active this frame...
            if flank1R.status == STARTED:
                # update params
                pass
            
            # if flank1R is stopping this frame...
            if flank1R.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > flank1R.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    flank1R.tStop = t  # not accounting for scr refresh
                    flank1R.tStopRefresh = tThisFlipGlobal  # on global time
                    flank1R.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'flank1R.stopped')
                    # update status
                    flank1R.status = FINISHED
                    flank1R.setAutoDraw(False)
            
            # *flank2R* updates
            
            # if flank2R is starting this frame...
            if flank2R.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                flank2R.frameNStart = frameN  # exact frame index
                flank2R.tStart = t  # local t and not account for scr refresh
                flank2R.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(flank2R, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'flank2R.started')
                # update status
                flank2R.status = STARTED
                flank2R.setAutoDraw(True)
            
            # if flank2R is active this frame...
            if flank2R.status == STARTED:
                # update params
                pass
            
            # if flank2R is stopping this frame...
            if flank2R.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > flank2R.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    flank2R.tStop = t  # not accounting for scr refresh
                    flank2R.tStopRefresh = tThisFlipGlobal  # on global time
                    flank2R.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'flank2R.stopped')
                    # update status
                    flank2R.status = FINISHED
                    flank2R.setAutoDraw(False)
            
            # *flank3R* updates
            
            # if flank3R is starting this frame...
            if flank3R.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                flank3R.frameNStart = frameN  # exact frame index
                flank3R.tStart = t  # local t and not account for scr refresh
                flank3R.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(flank3R, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'flank3R.started')
                # update status
                flank3R.status = STARTED
                flank3R.setAutoDraw(True)
            
            # if flank3R is active this frame...
            if flank3R.status == STARTED:
                # update params
                pass
            
            # if flank3R is stopping this frame...
            if flank3R.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > flank3R.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    flank3R.tStop = t  # not accounting for scr refresh
                    flank3R.tStopRefresh = tThisFlipGlobal  # on global time
                    flank3R.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'flank3R.stopped')
                    # update status
                    flank3R.status = FINISHED
                    flank3R.setAutoDraw(False)
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_2 is stopping this frame...
            if key_resp_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_2.tStop = t  # not accounting for scr refresh
                    key_resp_2.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_2.stopped')
                    # update status
                    key_resp_2.status = FINISHED
                    key_resp_2.status = FINISHED
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['a','l'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_2.keys == str(correctAns)) or (key_resp_2.keys == correctAns):
                        key_resp_2.corr = 1
                    else:
                        key_resp_2.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Trial" ---
        for thisComponent in Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Trial
        Trial.tStop = globalClock.getTime(format='float')
        Trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Trial.stopped', Trial.tStop)
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
            # was no response the correct answer?!
            if str(correctAns).lower() == 'none':
               key_resp_2.corr = 1;  # correct non-response
            else:
               key_resp_2.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp_2.keys',key_resp_2.keys)
        trials.addData('key_resp_2.corr', key_resp_2.corr)
        if key_resp_2.keys != None:  # we had a response
            trials.addData('key_resp_2.rt', key_resp_2.rt)
            trials.addData('key_resp_2.duration', key_resp_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Trial.maxDurationReached:
            routineTimer.addTime(-Trial.maxDuration)
        elif Trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.700000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "iti" ---
    # create an object to store info about Routine iti
    iti = data.Routine(
        name='iti',
        components=[text_2],
    )
    iti.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for iti
    iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    iti.tStart = globalClock.getTime(format='float')
    iti.status = STARTED
    thisExp.addData('iti.started', iti.tStart)
    iti.maxDuration = None
    # keep track of which components have finished
    itiComponents = iti.components
    for thisComponent in iti.components:
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
    
    # --- Run Routine "iti" ---
    thisExp.currentRoutine = iti
    iti.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + itiDur-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=iti,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            iti.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if iti.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in iti.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "iti" ---
    for thisComponent in iti.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for iti
    iti.tStop = globalClock.getTime(format='float')
    iti.tStopRefresh = tThisFlipGlobal
    thisExp.addData('iti.stopped', iti.tStop)
    thisExp.nextEntry()
    # the Routine "iti" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_end" ---
    # create an object to store info about Routine practice_end
    practice_end = data.Routine(
        name='practice_end',
        components=[text_3, key_resp_3],
    )
    practice_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for practice_end
    practice_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_end.tStart = globalClock.getTime(format='float')
    practice_end.status = STARTED
    thisExp.addData('practice_end.started', practice_end.tStart)
    practice_end.maxDuration = None
    # keep track of which components have finished
    practice_endComponents = practice_end.components
    for thisComponent in practice_end.components:
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
    
    # --- Run Routine "practice_end" ---
    thisExp.currentRoutine = practice_end
    practice_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=practice_end,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            practice_end.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if practice_end.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in practice_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_end" ---
    for thisComponent in practice_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_end
    practice_end.tStop = globalClock.getTime(format='float')
    practice_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_end.stopped', practice_end.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "practice_end" was not non-slip safe, so reset the non-slip timer
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
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


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
    logging.flush()
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
