import torch
import numpy as np
from tqdm import tqdm

from Common import ConstVar
from DeepLearning import utils


class Tester:
    def __init__(self, modelG, modelD, metric_fn, test_dataloader, device):
        """
        * 테스트 관련 클래스
        :param modelG: 테스트 할 모델. 생성자
        :param modelD: 테스트 할 모델. 판별자
        :param metric_fn: 학습 성능 체크하기 위한 metric
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.modelG = modelG
        self.modelD = modelD
        # 학습 성능 체크하기 위한 metric
        self.metric_fn = metric_fn
        # 테스트용 데이터로더
        self.test_dataloader = test_dataloader
        # GPU / CPU
        self.device = device

    def running(self, sample_z_collection=None, checkpoint_file=None):
        """
        * 테스트 셋팅 및 진행
        :param sample_z_collection: 생성자로 이미지를 생성하기 위한 샘플 noise z 모음
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 테스트 수행됨
        """

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.modelG.load_state_dict(state[ConstVar.KEY_STATE_MODEL_G])
            self.modelD.load_state_dict(state[ConstVar.KEY_STATE_MODEL_D])

        # 테스트 진행
        self._test(sample_z_collection=sample_z_collection)

    def _test(self, sample_z_collection):
        """
        * 테스트 진행
        :param sample_z_collection: 생성자로 이미지를 생성하기 위한 샘플 noise z 모음
        :return: 이미지 생성 및 score 기록
        """

        # 각 모델을 테스트 모드로 전환
        self.modelG.eval()
        self.modelD.eval()

        # 배치 마다의 BCE loss 담을 리스트
        batch_bce_loss_listG = list()
        batch_bce_loss_listD = list()

        # 생성된 이미지 담을 리스트
        self.pics_list = list()

        for x in tqdm(self.test_dataloader, desc='test dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = x.shape[0]

            # real image label
            real_label = torch.ones(batch_size, device=self.device)
            # fake image label
            fake_label = torch.zeros(batch_size, device=self.device)

            # noise z
            z = torch.randn(size=(batch_size, 100, 1, 1), device=self.device)

            # 텐서를 해당 디바이스로 이동
            x = x.to(self.device)

            # 판별자 real image 순전파
            output = self.modelD(x)
            scoreD_real = self.metric_fn(output=output,
                                         label=real_label)
            # 판별자 fake image 순전파
            fake_x = self.modelG(z)
            output = self.modelD(fake_x)
            scoreD_fake = self.metric_fn(output=output,
                                         label=fake_label)
            # 배치 마다의 판별자 BCE loss 계산
            scoreD = scoreD_real + scoreD_fake
            batch_bce_loss_listD.append(scoreD)

            # 생성자 순전파
            fake_x = self.modelG(z)
            output = self.modelD(fake_x)
            # 배치 마다의 생성자 BCE loss 계산
            scoreG = self.metric_fn(output=output,
                                    label=real_label)
            batch_bce_loss_listG.append(scoreG)

        if sample_z_collection is not None:
            # 샘플 noise z 모음으로 이미지 생성하기. deepcopy 오류 방지를 위해 detach
            generated_image_collection = self.modelG(sample_z_collection).detach()
            self.pics_list = generated_image_collection

        # score 기록
        self.score = {
            ConstVar.KEY_SCORE_G: np.mean(batch_bce_loss_listG),
            ConstVar.KEY_SCORE_D: np.mean(batch_bce_loss_listD)
        }
