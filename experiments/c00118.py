import logging
import h5py
import pandas as pd
import time
from epics import caget, caget_many, PV
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from tmo.db import camviewer
from pcdsdevices.areadetector import plugins
from pcdsdevices.slits import PowerSlits
from pcdsdevices.pim import PPM
from ophyd import EpicsSignalRO
from ophyd import Component as Cpt
from bluesky.plans import scan, grid_scan, list_scan
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

# Simulation tools
from ophyd.positioner import SoftPositioner
from ophyd.sim import motor1, motor2, det1
import bluesky.plan_stubs as bps

logger = logging.getLogger(__name__)

PPMS = [
    'im2k4',
    'im3k4',
    'im4k4',
    'im5k4'
]

PPM_EXT = ':PPM:SPM:VOLT_BUFFER_RBV'

CAMS = [
    'im1k0',
    'im2k0',
    'im1k4',
    'im2k4',
    'im3k4',
    'im4k4',
    'im5k4'
]

EPICSARCH = [
    'EM2K0:XGMD:HPS:KeithleySum',
    'IM2K4:PPM:SPM:VOLT_RBV',
    'IM3K4:PPM:SPM:VOLT_RBV',
    'IM4K4:PPM:SPM:VOLT_RBV',
    'IM5K4:PPM:SPM:VOLT_RBV',
    'MR1K4:SOMS:MMS:XUP.RBV',
    'MR1K4:SOMS:MMS:YUP.RBV',
    'MR1K4:SOMS:MMS:PITCH.RBV',
    'MR2K4:KBO:MMS:X.RBV',
    'MR2K4:KBO:MMS:Y.RBV',
    'MR2K4:KBO:MMS:PITCH.RBV',
    'MR2K4:KBO:MMS:BEND:US.RBV',
    'MR2K4:KBO:MMS:BEND:DS.RBV',
    'MR3K4:KBO:MMS:X.RBV',
    'MR3K4:KBO:MMS:Y.RBV',
    'MR3K4:KBO:MMS:PITCH.RBV',
    'MR3K4:KBO:MMS:BEND:US.RBV',
    'MR3K4:KBO:MMS:BEND:DS.RBV',
    'PF1K4:WFS:MMS:Y.RBV',
    'PF1K4:WFS:MMS:Z.RBV'
]

GMD = 'EM1K0:GMD:HPS:AvgPulseIntensity'
XGMD = 'EM2K0:XGMD:HPS:AvgPulseIntensity'

PGMD = 'EM1K0:GMD:HPS:milliJoulesPerPulse'
PXGMD = 'EM2K0:XGMD:HPS:milliJoulesPerPulse'

PPM_PATH = '/cds/home/opr/tmoopr/experiments/x43218/ppm_data/'
CAM_PATH = '/cds/home/opr/tmoopr/experiments/x43218/cam_data/'
SCAN_PATH = '/cds/home/opr/tmoopr/experiments/x43218/scan_data/'

class PPMRecorder:
    _data = []
    _gmd_data = []
    _xgmd_data = []
    _ppm = 'im2k4'
    _collection_time = 60

    @staticmethod
    def ppms():
        return PPMS

    @property
    def ppm(self):
        """Name of ppm"""
        return self._ppm

    @ppm.setter
    def ppm(self, ppm):
        self._ppm = ppm

    @property
    def collection_time(self):
        """Time for collection"""
        return self._collection_time

    @collection_time.setter
    def collection_time(self, ct):
        try:
            self._collection_time = float(ct)
        except:
            logger.warning('collection time must be number')

    @property
    def data(self):
        return self._data

    @property
    def gmd_data(self):
        return self._gmd_data

    @property
    def xgmd_data(self):
        return self._xgmd_data

    def clear_data(self):
        self._data = []
        self._gmd_data = []
        self._xgmd_data = []

    def collect(self):
        if self.data: 
            logger.info('Found leftover data, clearing')
            self.clear_data()

        logger.info(f'Starting PPM scan for {self.ppm}')
        ppm_obj = EpicsSignalRO(self.ppm.upper()+PPM_EXT)
        #print('getting ', ppm_obj.get())
        ppm_obj.wait_for_connection(timeout=1.0) 
        uid = ppm_obj.subscribe(self.data_cb)
        time.sleep(self.collection_time)  # Threading?
        logger.info('Done collecting PPM data')
        ppm_obj.unsubscribe(uid)

    def data_cb(self, value, **kwargs):
        """Collect all the data"""
        self.data.extend(value)
        self.gmd_data.append(caget(GMD))
        self.xgmd_data.append(caget(XGMD))

    def downsample(self, downsample=100, ave=True):
        """General method for downsampling in even intervals, could be faster"""
        if not self.data:
            logger.warning('Trying to downsample empty dataset')
            return
        logger.info(f'Downsampling data by a factor of {downsample}') 
        if ave:
            segments = range(int(len(self.data) / downsample))
            self._data = [np.mean(self.data[i*downsample:(i+1)*downsample]) for i in segments]
        else:
            self._data = self.data[::downsample]

    def plot(self):
        plt.title(f'time plot of {self.ppm}')
        plt.plot(self.data)

    def save_hdf5(self, file_name=None):
        if not file_name:
            file_name = f'{self.ppm}-{int(time.time())}.h5'
        location = ''.join([PPM_PATH, file_name])
        hf = h5py.File(location, 'w')
        hf.create_dataset('ppm_data', data=self.data)
        hf.create_dataset('gmd_data', data=self._gmd_data)
        hf.create_dataset('xgmd_data', data=self._xgmd_data)
        hf.close()
        logger.info(f'wrote all data to {location}')

class CamTools:
    _camera = camviewer.im1k0
    _cam_type = 'opal'
    _path = CAM_PATH
    _images = []
    _timestamps = []
    _cb_uid = None
    _num_images = 10
    _state = 'idle'

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, camera):
        try:
            self._camera = getattr(camviewer, camera)
        except AttributeError as e:
            logger.warning(f'{camera} is not a valid camera: {e}')

    @property
    def height(self):
        if self.camera:
            return self.camera.image2.height.get()
        else:
            return None

    @property
    def width(self):
        if self.camera:
            return self.camera.image2.width.get()
        else:
            return None

    @property
    def file_path(self):
        return self._path

    @file_path.setter
    def file_path(self, path):
        # We'll have to validate or make path
        self._path = path

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def images(self):
        return self._images

    @property
    def num_images(self):
        return self._num_images

    @num_images.setter
    def num_images(self, num):
        try:
            self._num_images = int(num)
        except:
            logger.warning('number of images must be castable as int')

    @staticmethod
    def camera_names():
        return CAMS

    def clear_data(self):
        self._images = []
        self._timestamps = []

    def collect(self, n_img):
        """General collection method.  If n_img specified, set
        property as well"""
        if not self.num_images:
            if n_img:
                self.num_images = n_img
            else:
                logger.warning('You need to specify number of images to collect')

        if self.images:
            logger.info('Leftover image data, clearing')
            self._images = []
            self._timestamps = []

        if not self.camera:
            logger.warning('You have not specified a camera')        
            return

        #if self.camera.cam.acquire.get() is not 1:
        #    logger.info('Camera has no rate, starting acquisition')
        #    self.camera.cam.acquire.put(1)

        cam_model = self.camera.cam.model.get()
        # TODO: Make dir with explicit cam model
        if 'opal' in cam_model:
            self._cam_type = 'opal'
        else:
            self._cam_type = 'gige'
        
        logger.info(f'Starting data collection for {self.camera.name}')
        #self._cb_uid = self.camera.image2.array_data.subscribe(self._data_cb)
        self._get_data()

    def _get_data(self):
        delay = 0.1
        while len(self.images) < self.num_images:
            img = self.camera.image2.array_data.get()
            ts = self.camera.image2.array_data.timestamp
            if len(self.images) == 0:
                self.images.append(np.reshape(img, (self.height, self.width)))
                self.timestamps.append(ts)
            if not np.array_equal(self.images[-1], img):
                print('getting image: ', len(self.images))
                self.images.append(np.reshape(img, (self.height, self.width)))
                self.timestamps.append(ts)
                print('delay ', time.time() - ts)
                time.sleep(delay)
            else:
                time.sleep(0.01)   
        logger.info('done collecting image data ')     

    def _data_cb(self, **kwargs):
        """Area detector cbs does not know thyself"""
        obj = kwargs.get('obj')
        arr = obj.value
        ts = obj.timestamp
        self.images.append(np.reshape(arr, (self.height, self.width)))
        self.timestamps.append(ts)
        logger.info('received image: ', len(self.images), time.time() - ts)
        if len(self.images) >= self.num_images:
            logger.info('We have collected all our images, stopping collection')
            self.camera.image2.array_data.unsubscribe(self._cb_uid)

    def plot(self):
        """Let people look at collected images"""
        if not self.images:
            info.warning('You do not have any images collected')

        num_images = len(self.images)
        img_sum = self.images[0]
        if num_images is 1:
            plt.imshow(img_sum)
        else:
            for img in self.images[1:]:
                img_sum += img
            plt.imshow(img_sum / num_images)

    def save(self):
        file_name = f'{self.camera.name}-{int(time.time())}.h5'
        location = ''.join([self._path, self._cam_type, '/', file_name])
        hf = h5py.File(location, 'w')
        hf.create_dataset('image_data', data=self.images)
        hf.create_dataset('timestamps', data=self.timestamps)
        hf.close()
        logger.info(f'wrote all image data to {location}')
        return location
#class SimPlans:


class SimEvrScan:
    _motor = motor1
    _evr = SoftPositioner(name='EVR:TDES')
    _RE = RunEngine({})
    _scan_id = None
    
    @property
    def scan_id(self):
        return self._scan_id

    @scan_id.setter
    def scan_id(self, uid):
        self._scan_id = uid

    def start(self, evr_start, evr_stop, evr_steps, motor_start, motor_stop, motor_steps):
        """Set TDES, then scan the x motor"""
        return grid_scan([det1],
                         self._evr, evr_start, evr_stop, evr_steps,
                         self._motor, motor_start, motor_stop, motor_steps)

class User:
    _cam_tools = CamTools()    
#    _sim_cs = SimEvrScan()
    _fee_slit = PowerSlits(name='sl2k0', prefix='SL2K0:POWER')

    @staticmethod
    def ppm_scan(ppm='im2k4', time=60, downsample=100, plot=True, save=True):
        """General PPM Recorder Scan with basic steps that Phil uses

        Parameters:
        ----------
        ppm: str <default: 'im2k4'>
            The root name of the PPM you'd like to record for

        time: int <default: 60>
            The amount of time you would like to record for in seconds

        downsample: int <default: 100>
            The amount you would like to downsample.  Averages points
            in between

        plot: bool <default: True>
            If you'd like to have show a plot of the PPM data

        save: bool <default: True>
            If you'd like to save the data to default location
        """
        recorder = PPMRecorder()
        recorder.ppm = ppm
        recorder.collection_time = time
        recorder.collect()
        if plot:
            recorder.plot()

        if downsample > 1:
            recorder.downsample(downsample=downsample)

        if save:
            recorder.save_hdf5()

    @property
    def cam_tools(self):
        return self._cam_tools

    @property
    def sl2k0(self):
        return self._fee_slit

    @property
    def slits(self):
        return ['top', 'bottom', 'north', 'south']

    def slit_scan(self, slit_name):
        if slit not in self.slits:
            info.warning(f'{slit} not in list of slits, exiting')
            return
        
        # BeckhoffAxisTuple
        slit = getattr(self.sl2k0, slit_name)
        
    #def slit_scan(self, 

    def scanner(self, mov_pv, start, stop, steps, tol, cam, images=10, rbck_ext='.RBV'):
        """General scanner for now because we want to take images"""
        steps = np.linspace(start, stop, steps)
        times = []
        image_files = []
        self.cam_tools.camera = cam
        try:
            pv_obj = PV(mov_pv)
            pv_rbck = PV(mov_pv + rbck_ext)
        except:
            logger.warning('Unable to connect to {mov_pv}')
        df = pd.DataFrame()
        for step in steps:
            pv_obj.put(step)
            while abs(pv_rbck.get() - step) > tol:
                time.sleep(0.1)
            logger.info(f'Stepper reached {step}, collecting data')
            times.append(time.time())
            self.cam_tools.collect(images)
            f = self.cam_tools.save()
            image_files.append(f)
            df = df.append(pd.DataFrame([caget_many(EPICSARCH)], columns=EPICSARCH), ignore_index=True)
        df = df.assign(times=times)
        df = df.assign(image_files=image_files)
        file_name = f'{mov_pv}-{int(time.time())}.h5'
        location = SCAN_PATH + file_name
        df.to_hdf(location, key='metadata')
        logger.info(f'wrote all data to {location}')
#    @property
#    def sim_crazy_scan(self):
#        return self._sim_cs
