import unittest

from rfm_mfg_stationary.mfg_1d.utils.utils_1d import init_rfm, set_seed


class MyTestCase(unittest.TestCase):
    def test_stationary_fokker_planck(self):
        """
        We test the fokker-planck equation solver with Ornstein-Uhlenbeck Process
        :return:
        """
        set_seed(100)
        M_p_fp, J_n_fp, Q_fp = 4, 100, 100
        models_fp, collocs_fp = init_rfm(M_p_fp, J_n_fp, Q_fp)


if __name__ == '__main__':
    unittest.main()
