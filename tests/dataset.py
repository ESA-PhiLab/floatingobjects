import os
import sys
this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder, "..", "code"))

import unittest
from data import FloatingSeaObjectRegionDataset, FloatingSeaObjectDataset
import numpy as np

TESTS_DATA_ROOT = os.environ.get('TESTS_DATA_ROOT', '/data')

class DatasetRegionTestCases(unittest.TestCase):

    def eval_region(self, region):
        ds = FloatingSeaObjectRegionDataset(TESTS_DATA_ROOT, region)
        for x, y, fid in ds:
            self.assertTrue(isinstance(fid, str))
            self.assertTrue(isinstance(x, np.ndarray))
            self.assertTrue(isinstance(y, np.ndarray))
            self.assertTrue(len(x.shape) == 3)  # D x H x W
            self.assertTrue(len(y.shape) == 2)  # H x W
            self.assertTrue(x.shape[1] == y.shape[0])  # H == H
            self.assertTrue(x.shape[2] == y.shape[1])  # W == W

    def test_danang_20181005(self):
        self.eval_region("danang_20181005")

    def test_biscay_20180419(self):
        self.eval_region("biscay_20180419")

    def test_lagos_20200505(self):
        self.eval_region("lagos_20200505")

    def test_riodejaneiro_20180504(self):
        self.eval_region("riodejaneiro_20180504")

    def test_toledo_20191221(self):
        self.eval_region("toledo_20191221")

    def test_longxuyen_20181102(self):
        self.eval_region("longxuyen_20181102")

    def test_mandaluyong_20180314(self):
        self.eval_region("mandaluyong_20180314")

    def test_panama_20190425(self):
        self.eval_region("panama_20190425")

    def test_sandiego_20180804(self):
        self.eval_region("sandiego_20180804")

    def test_vungtau_20180423(self):
        self.eval_region("vungtau_20180423")

    def test_venice_20180928(self):
        self.eval_region("venice_20180928")

    def test_accra_20181031(self):
        self.eval_region("accra_20181031")

    def test_lagos_20190101(self):
        self.eval_region("lagos_20190101")

    def test_venice_20180630(self):
        self.eval_region("venice_20180630")

    def test_kolkata_20201115(self):
        self.eval_region("kolkata_20201115")

    def test_kentpointfarm_20180710(self):
        self.eval_region("kentpointfarm_20180710")

    def test_tungchungChina_20190922(self):
        self.eval_region("tungchungChina_20190922")

    def test_london_20180611(self):
        self.eval_region("london_20180611")

    def test_neworleans_20200202(self):
        self.eval_region("neworleans_20200202")

    def test_portalfredSouthAfrica_20180601(self):
        self.eval_region("portalfredSouthAfrica_20180601")

    def test_shengsi_20190615(self):
        self.eval_region("shengsi_20190615")

    def test_suez_20200403(self):
        self.eval_region("suez_20200403")

    def test_tangshan_20180130(self):
        self.eval_region("tangshan_20180130")

    def test_tunisia_20180715(self):
        self.eval_region("tunisia_20180715")

    def test_turkmenistan_20181030(self):
        self.eval_region("turkmenistan_20181030")

    def test_sanfrancisco_20190219(self):
        self.eval_region("sanfrancisco_20190219")


class DatasetTestCases(unittest.TestCase):

    def eval_fold(self, seed):
        train = FloatingSeaObjectDataset(TESTS_DATA_ROOT, fold="train", seed=seed)
        val = FloatingSeaObjectDataset(TESTS_DATA_ROOT, fold="val", seed=seed)
        test = FloatingSeaObjectDataset(TESTS_DATA_ROOT, fold="test", seed=seed)

        # check if regions have any intersections (same regions in both train and test)
        self.assertTrue(len(np.intersect1d(train.regions, val.regions)) == 0)
        self.assertTrue(len(np.intersect1d(val.regions, test.regions)) == 0)
        self.assertTrue(len(np.intersect1d(train.regions, test.regions)) == 0)

        # get some data
        self.assertTrue(len(train[0]) == 3)
        self.assertTrue(len(val[0]) == 3)
        self.assertTrue(len(test[0]) == 3)

    def test_trainvalsplit_fold0(self):
        self.eval_fold(0)

    def test_trainvalsplit_fold1(self):
        self.eval_fold(1)

    def test_trainvalsplit_fold2(self):
        self.eval_fold(2)

    def test_trainvalsplit_fold3(self):
        self.eval_fold(3)

    def test_trainvalsplit_fold4(self):
        self.eval_fold(4)


if __name__ == '__main__':
    unittest.main()