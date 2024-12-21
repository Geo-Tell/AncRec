import numpy as np

class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N, mode):
        self.N = N
        self.mode = mode

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data['points']
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            if self.mode == 'test':
                idx = np.arange(0, self.N)
            else:
                idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                'points': points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            if self.mode == 'test':
                idx0 = np.arange(0, Nt_out)
                idx1 = np.arange(0, Nt_in)
            else:
                idx0 = np.random.randint(points0.shape[0], size=Nt_out)
                idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                'points': points,
                'occ': occ,
                'volume': volume,
            })
        return data_out


class SubsamplePoints2(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N, mode, occ_threshold = 0.5, sdf_threshold = 0.0, use_sdf = True):
        self.N = N
        self.mode = mode
        self.occ_threshold = occ_threshold
        self.sdf_threshold = sdf_threshold
        self.use_sdf = use_sdf


    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data['points']
        occ = data['occ']
        sdf = data['sdf']

        data_out = data.copy()
        if isinstance(self.N, int):
            if self.mode == 'test':
                idx = np.arange(0, self.N)
            else:
                idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                'points': points[idx, :],
                'occ':  occ[idx],
                'sdf' :sdf[idx]
            })
        else:
            Nt_out, Nt_in = self.N
            if self.use_sdf:
                sdf_binary = (sdf <= self.sdf_threshold)
                points0 = points[~sdf_binary] #外
                points1 = points[sdf_binary]  #内
                sdf0 = sdf[~sdf_binary]
                sdf1 = sdf[sdf_binary]
            else:
                occ_binary = (occ >= self.occ_threshold)
                points0 = points[~occ_binary]
                points1 = points[occ_binary]
                sdf0 = sdf[~occ_binary]
                sdf1 = sdf[occ_binary]

            if self.mode == 'test':
                idx0 = np.arange(0, Nt_out)
                idx1 = np.arange(0, Nt_in)
            else:
                idx0 = np.random.randint(points0.shape[0], size=Nt_out)
                idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)


            sdf0 = sdf0[idx0]
            sdf1 = sdf1[idx1]
            sdf = np.concatenate([sdf0, sdf1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32) #< threshold: out
            occ1 = np.ones(Nt_in, dtype=np.float32)   # >= threshold: in
            occ = np.concatenate([occ0, occ1], axis=0)


            if self.use_sdf:
                volume = sdf_binary.sum() / len(sdf_binary)
            else:
                volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                'points': points,
                'occ': occ,
                'sdf': sdf,
                'volume': volume,
            })
        return data_out