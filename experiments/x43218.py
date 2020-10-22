import logging
import h5py
import time
from epics import caget
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from tmo.db import camviewer
from pcdsdevices.areadetector import plugins
from pcdsdevices.slits import PowerSlits
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
    'IM1K3:PPM:SPM:VOLT_BUFFER_RBV',
    'IM2K4:PPM:SPM:VOLT_BUFFER_RBV'
]

CAMS = [
    'im1k0',
    'im2k0',
    'im1k4',
    'im2k4',
    'im3k4',
    'im4k4',
    'im5k4'
]

GMD = 'EM1K0:GMD:HPS:AvgPulseIntensity'
XGMD = 'EM2K0:XGMD:HPS:AvgPulseIntensity'

PGMD = 'EM1K0:GMD:HPS:milliJoulesPerPulse'
PXGMD = 'EM2K0:XGMD:HPS:milliJoulesPerPulse'

PPM_PATH = '/cds/home/opr/tmoopr/experiments/x43218/ppm_data/'
CAM_PATH = '/cds/home/opr/tmoopr/experiments/x43218/cam_data/'

class PPMRecorder:
    _data = []
    _gmd_data = []
    _xgmd_data = []
    _ppm_pv = 'IM2K4:PPM:SPM:VOLT_BUFFER_RBV'
    _collection_time = 1

    @staticmethod
    def ppm_pvs():
        return PPMS

    @property
    def ppm_pv(self):
        """PPM pv to get data from"""
        return self._ppm_pv

    @ppm_pv.setter
    def ppm_pv(self, ppm_pv):
        self._ppm_pv = ppm_pv

    @property
    def collection_time(self):
        """Time for collection"""
        return self._collection_time

    @collection_time.setter
    def collection_time(self, ct):
        if not isinstance(ct, int) and not isinstance(ct, float):
            logger.warning('collection time must be a number')
        
        self._collection_time = ct

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, list):
            logger.warning('Data must be a list')
            return

        self._data = data

    @property
    def gmd_data(self):
        return self._gmd_data

    @property
    def xgmd_data(self):
        return self._xgmd_data

    def clear_data(self):
        self.data = []
        self._gmd_data = []
        self._xgmd_data = []

    def _connect(self):
        try:
            pv_obj = EpicsSignalRO(self.ppm_pv)
            pv_obj.wait_for_connection(timeout=1.0)
            return pv_obj
        except TimeoutError as e:
            logger.warning(f'Unable to connect to {self.ppm_pv}')
            return None

    def collect(self):
        if self.ppm_pv:
            pv_obj = self._connect()
            if not pv_obj:
                return  # This implementation needs to change
        else:
            logger.warning('You must provide a ppm pv to collect')
            return
        if self.data: 
            logger.info('Found leftover data, clearing')
            self.clear_data()

        uid = pv_obj.subscribe(self.data_cb)
        time.sleep(self.collection_time)  # Threading?
        pv_obj.unsubscribe(uid)
        logger.info('Done collecting PPM data')

    def data_cb(self, value, **kwargs):
        """Collect all the data"""
        self.data.extend(value)
        self.gmd_data.append(caget(GMD))
        self.xgmd_data.append(caget(XGMD))

    def downsample(self, downsample=10, ave=True):
        """General method for downsampling in even intervals, could be faster"""
        if not self.data:
            log.warning('Trying to downsample empty dataset')
            return
        
        if ave:
            segments = range(int(len(self.data) / downsample))
            self.data = [np.mean(self.data[i*downsample:(i+1)*downsample]) for i in segments]
        else:
            self.data = self.data[::downsample]

    def save_data(self, file_name=None):
        if not file_name:
            file_name = f'{self.ppm_pv}-{int(time.time())}.data'
        location = ''.join([PPM_PATH, file_name])
        with open(location, 'w') as f:
            for val in self.data:
                f.write(str(val))
        logger.info(f'wrote all data to {location}')  

    def plot(self):
        plt.title(f'time plot of {self.ppm_pv}')
        plt.plot(self.data)

    def save_hdf5(self, file_name=None):
        if not file_name:
            file_name = f'{self.ppm_pv}-{int(time.time())}.h5'
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
    _cb_uid = None
    _num_images = 10

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
    def images(self):
        return self._images

    @images.setter
    def images(imgs, self):
        if not isinstance(imgs, np.ndarray):
            logger.warning('images must be in np.ndarray')
            return

        self._images = imgs

    @property
    def num_images(self):
        return self._num_images

    @num_images.setter
    def num_images(self, num):
        try:
            self._num_images = int(num)
        except:
            logger.warning('number of images must be able to cast as int')

    @staticmethod
    def camera_names():
        return CAMS

    def collect(self, n_img=10):
        if self.images:
            logger.info('Leftover image data, clearing')

        if not self.camera:
            logger.warning('You have not specified a camera')        
            return

        if self.camera.cam.acquire.get() is not 1:
            logger.info('Camera has no rate, starting acquisition')
            self.camera.cam.acquire.put(1)

        cam_model = self.camera.cam.model.get()
        # TODO: Make dir with explicit cam model
        if 'opal' in cam_model:
            self._cam_type = 'opal'
        else:
            self._cam_type = 'gige'
        
        logger.info(f'Starting data collection for {self.camera.name}')
        self._cb_uid = self.camera.image2.array_data.subscribe(self._data_cb)

    def _data_cb(self, **kwargs):
        """Area detector cbs does not know thyself"""
        arr = kwargs.get('obj').value
        self.images.append(np.reshape(arr, (self.height, self.width)))
        print('got an image ', len(self.images))
        if len(self.images) == self.num_images:
            logger.info('We have collected all our images')
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
        hf.close()
        logger.info(f'wrote all image data to {location}')
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
    _ppm_recorder = PPMRecorder()
    _cam_tools = CamTools()    
#    _sim_cs = SimEvrScan()
    _fee_slit = PowerSlits(name='sl2k0', prefix='SL2K0:POWER')

    @property
    def ppm_recorder(self):
        return self._ppm_recorder

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

#    @property
#    def sim_crazy_scan(self):
#        return self._sim_cs
