import numpy as np
from scipy.io import loadmat

import aslam_cv as acv
import sm

# copied from https://github.com/JzHuai0108/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_common/VimapCsvReader.py
class FrameObservation(object):
    def __init__(self):
        self._stamp = acv.Time(0.0)
        self._T_t_c = sm.Transformation()
        self.landmarkList = [] # observed landmarks
        self.landmarkIdList = []
        self.observationList = [] # landmark observations in image

    def T_t_c(self):
        """
        :return: transform from camera to target
        """
        return self._T_t_c

    def set_T_t_c(self, T_t_c):
        self._T_t_c = T_t_c

    def time(self):
        return self._stamp

    def setTime(self, time):
        self._stamp = time

    def getCornersImageFrame(self):
        return self.observationList

    def getCornersTargetFrame(self):
        return self.landmarkList

    def getCornersIdx(self):
        return self.landmarkIdList

    def hasSuccessfulObservation(self):
        return len(self.observationList) > 0

    def getCornerReprojection(self, cameraGeometry):
        """

        :param cameraGeometry: eg, DistortedPinholeCameraGeometry
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def landmarkSentinel():
        return np.array([0, 0, 1e8])

    def removeUnusedKeypoints(self):
        keep = []

        for landmark in self.landmarkList:
            if np.allclose(landmark, FrameObservation.landmarkSentinel()):
                keep.append(False)
            else:
                keep.append(True)

        self.landmarkList = [landmark for index, landmark in enumerate(self.landmarkList) if keep[index]]
        self.landmarkIdList = [id for index, id in enumerate(self.landmarkIdList) if keep[index]]
        self.observationList = [observation for index, observation in enumerate(self.observationList) if keep[index]]

    def appendLandmark(self, landmark, landmarkId):
        self.landmarkList.append(landmark)
        self.landmarkIdList.append(landmarkId)

    def appendObservation(self, observation):
        self.observationList.append(observation)

    def __str__(self):
        header = "{:.9f} {} {}\n".format(self._stamp.toSec(), ' '.join(map(str, self._T_t_c.t())),
                                        ' '.join(map(str, sm.quatInv(self._T_t_c.q()))))
        msg = header
        for index, landmark in enumerate(self.landmarkList):
            msg += '{}: {} {}\n'.format(self.landmarkIdList[index], landmark, self.observationList[index])
        return msg


class MatImageDatasetIterator(object):
  def __init__(self, dataset, indices=None):
    self.dataset = dataset
    if indices is None:
      self.indices = np.arange(dataset.numImages())
    else:
      self.indices = indices
    self.iter = self.indices.__iter__()

  def __iter__(self):
    return self

  def __next__(self):
    idx = next(self.iter)
    return self.dataset.getImage(idx)

  next = __next__  # Python 2


def getObjectAndImagePointsWithStatus(board, corners):
    opts = []
    ipts = []
    times = []
    usedstatus = []
    boardpts = board['boards'][0, 0]['X']  # 2 x M
    numviews = corners['corners'].shape[1]
    imgsize = corners['imgsize'][0]
    totalused = corners['used'][0, 0]
    for im in range(numviews):
        m = corners['corners'][0, im]['x'][0, 0].shape[1]
        if m > 0:
            objpts = np.zeros((m, 1, 3), np.float32)
            imgpts = np.zeros((m, 1, 2), np.float32)
            ids = corners['corners'][0, im]["cspond"][0, 0].astype(int)
            imgarr = corners['corners'][0, im]['x'][0, 0]

            for kp in range(m):
                imgpts[kp, 0, :] = imgarr[:, kp]
                pid = ids[0, kp] - 1
                objpts[kp, 0, :2] = boardpts[:, pid]
            opts.append(objpts)
            ipts.append(imgpts)
            time = corners['times'][0, im]
            times.append(time)
            used = corners['corners'][0, im]['used'][0, 0][0, 0]
            usedstatus.append(used)
    assert(np.sum(usedstatus) == totalused)
    return opts, ipts, times, imgsize, usedstatus


def getObjectAndImagePoints(board, corners):
    opts = []
    ipts = []
    times = []
    lmkids = []

    boardpts = board['boards'][0, 0]['X']  # 2 x M
    numviews = corners['corners'].shape[1]
    imgsize = corners['imgsize'][0]
    for im in range(numviews):
        m = corners['corners'][0, im]['x'][0, 0].shape[1]
        if m > 0:
            objpts = np.zeros((m, 3), np.float32)
            imgpts = np.zeros((m, 2), np.float32)
            ids = corners['corners'][0, im]["cspond"][0, 0].astype(int)
            imgarr = corners['corners'][0, im]['x'][0, 0]
            intids = []

            for kp in range(m):
                imgpts[kp, :] = imgarr[:, kp]
                pid = ids[0, kp] - 1
                objpts[kp, :2] = boardpts[:, pid]
                intids.append(pid)

            opts.append(objpts)
            ipts.append(imgpts)
            lmkids.append(intids)
            time = corners['times'][0, im]
            times.append(time)
    return opts, lmkids, ipts, times, imgsize


class MatImageDatasetReader(object):
    """
    read corners data from mat files saved by tartan calib
    Currently only support one camera.
    """
    def __init__(self, boardfile, cornersfile, camid, from_to=None):
        self.cornersfile = cornersfile
        self.camid = camid
        self.topic = '/cam{}/image_raw'.format(camid)
        self.from_to = from_to
        self.imgsize = None
        self.targetObservations = self.loadCorners(boardfile, cornersfile)
        self.indices = np.arange(len(self.targetObservations))
        if from_to:
            self.indices = self.truncateIndicesFromTime(self.indices, from_to)

    def loadCorners(self, boardmat, cornersmat):
        observations = []
        board = loadmat(boardmat)
        corners = loadmat(cornersmat)
        opts, lmkids, ipts, times, imgsize = getObjectAndImagePoints(board, corners)
        self.imgsize = (int(imgsize[0]), int(imgsize[1]))
        nframes = len(ipts)
        for j in range(nframes):
            observations.append(FrameObservation())
        for j, pts in enumerate(ipts):
            observations[j].setTime(acv.Time(times[j]))
            observations[j].landmarkIdList = lmkids[j]
            observations[j].landmarkList = opts[j]
            observations[j].observationList = pts
        return observations

    def getObservations(self, target):
        observations = []
        totalpts = target.size()
        for j, myobs in enumerate(self.targetObservations):
            gt = acv.GridCalibrationTargetObservation(target)
            gt.setTime(myobs.time())
            fakeimg = np.zeros(self.imgsize).astype("uint8")
            gt.setImage(fakeimg)
            for i, id in enumerate(myobs.landmarkIdList):
                imgpt = np.array(myobs.observationList[i])
                objpt = myobs.landmarkList[i]
                gt.updateImagePoint(int(id), imgpt)
                tgtpt = target.point(int(id))
                norm = np.linalg.norm(objpt - tgtpt)
                assert(norm < 1e-7)
            observations.append(gt)
        return observations

    def truncateIndicesFromTime(self, indices, bag_from_to):
        # get the timestamps
        timestamps = [observation.time().toSec() for observation in self.targetObservations]

        bagstart = min(timestamps)
        baglength = max(timestamps) - bagstart

        # some value checking
        if bag_from_to[0] >= bag_from_to[1]:
            raise RuntimeError("Bag start time must be bigger than end time.".format(bag_from_to[0]))
        if bag_from_to[0] < 0.0:
            sm.logWarn("Bag start time of {0} s is smaller 0".format(bag_from_to[0]))
        if bag_from_to[1] > baglength:
            sm.logWarn("Bag end time of {0} s is bigger than the total length of {1} s".format(
                bag_from_to[1], baglength))

        # find the valid timestamps
        valid_indices = []
        for idx, timestamp in enumerate(timestamps):
            if timestamp >= (bagstart + bag_from_to[0]) and timestamp <= (bagstart + bag_from_to[1]):
                valid_indices.append(idx)
        sm.logWarn(
            "VimapCsvReader: truncated {0} / {1} images.".format(len(indices) - len(valid_indices),
                                                                 len(indices)))
        return valid_indices

    def __iter__(self):
        # Reset the bag reading
        return self.readDataset()

    def readDataset(self):
        return MatImageDatasetIterator(self, self.indices)

    def numImages(self):
        return len(self.indices)

    def getImage(self, idx):
        return self.targetObservations[idx]

